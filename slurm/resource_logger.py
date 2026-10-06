import argparse
import csv
import os
import signal
import time

import psutil

# --- OPTIONAL IMPORTS (Lazy Loaded later) ---
# We do NOT import pandas or matplotlib here to keep memory footprint < 20MB.

# Try to import pynvml for GPU monitoring
try:
    from pynvml import (
        NVMLError,
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlInit,
        nvmlShutdown,
    )

    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="HPC Resource Monitor (RAM/VRAM)")
    parser.add_argument("--output", required=True, help="Path to save CSV results")
    parser.add_argument(
        "--interval", type=int, default=5, help="Sampling interval in seconds"
    )
    parser.add_argument(
        "--pid", type=int, required=True, help="Main Process ID to monitor"
    )
    parser.add_argument("--plot", action="store_true", help="Generate plot on exit")
    return parser.parse_args()


def get_process_tree_memory(parent_pid):
    """
    Computes the total Resident Set Size (RSS) of a PID and all its children.
    Returns memory in GB.
    """
    total_mem = 0.0
    try:
        parent = psutil.Process(parent_pid)
        # Memory of the parent process
        total_mem += parent.memory_info().rss

        # Memory of all child processes (recursive=True handles grandchildren)
        for child in parent.children(recursive=True):
            try:
                total_mem += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    except (psutil.NoSuchProcess, psutil.AccessDenied):
        # Process might have already finished
        return 0.0

    return total_mem / (1024**3)


def get_gpu_memory(device_count):
    """
    Returns a list of used VRAM in GB for each GPU.
    """
    gpu_mems = []
    if NVML_AVAILABLE and device_count > 0:
        try:
            for i in range(device_count):
                handle = nvmlDeviceGetHandleByIndex(i)
                info = nvmlDeviceGetMemoryInfo(handle)
                gpu_mems.append(info.used / (1024**3))
        except NVMLError:
            # If GPU query fails, return empty list or handle gracefully
            pass
    return gpu_mems


def generate_plot(csv_path):
    """
    Generates a PNG plot from the captured CSV data.
    Uses lazy importing to save memory during the main run.
    """
    print(f"[Logger] Attempting to generate plot for {csv_path}...")

    # --- LAZY IMPORT START ---
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("[Logger] Plotting skipped: pandas or matplotlib not installed.")
        return
    # --- LAZY IMPORT END ---

    try:
        # Read CSV
        try:
            df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            print("[Logger] CSV is empty. No data to plot.")
            return

        if df.empty or len(df) < 2:
            print("[Logger] Not enough data points to plot.")
            return

        # Calculate elapsed time
        start_time = df["timestamp"].iloc[0]
        df["elapsed_sec"] = df["timestamp"] - start_time

        # Create Figure
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot System RAM (Left Y-axis)
        line1 = ax1.plot(
            df["elapsed_sec"],
            df["cpu_ram_gb"],
            label="System RAM (Total Tree)",
            color="blue",
            linewidth=2,
        )
        ax1.set_xlabel("Elapsed Time (seconds)")
        ax1.set_ylabel("System RAM (GB)", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.grid(True, alpha=0.3)

        # Plot GPU VRAM (Right Y-axis)
        gpu_cols = [c for c in df.columns if c.startswith("gpu_")]
        lines = line1
        if gpu_cols:
            ax2 = ax1.twinx()
            for i, col in enumerate(gpu_cols):
                line = ax2.plot(
                    df["elapsed_sec"],
                    df[col],
                    label=f"VRAM ({col})",
                    linestyle="--",
                    alpha=0.7,
                )
                lines += line
            ax2.set_ylabel("GPU VRAM (GB)", color="black")
            # Align zero on both axes if possible, or leave automatic

        # Combined Legend
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc="upper left")

        title = f"Resource Usage Profile (PID: {os.getpid()})"
        plt.title(title)

        # Save Plot
        plot_path = csv_path.replace(".csv", ".png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"[Logger] Plot saved to: {plot_path}")
        plt.close(fig)  # Explicitly close to free memory

    except Exception as e:
        print(f"[Logger] Failed to generate plot: {e}")


def main():
    args = get_args()

    # 1. Setup GPU Monitoring
    device_count = 0
    if NVML_AVAILABLE:
        try:
            nvmlInit()
            device_count = nvmlDeviceGetCount()
            print(f"[Logger] NVML Initialized. Found {device_count} GPUs.")
        except NVMLError as e:
            print(f"[Logger] NVML Init failed: {e}")
    else:
        print("[Logger] pynvml not found. GPU monitoring disabled.")

    # 2. Setup Signal Handling (Graceful Exit)
    running = True

    def stop_handler(signum, frame):
        nonlocal running
        print(f"[Logger] Received signal {signum}. Stopping...")
        running = False

    # Capture standard termination signals
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)

    # 3. Open CSV and Start Logging
    print(
        f"[Logger] Monitoring PID {args.pid} (and children) every {args.interval}s..."
    )

    # Prepare CSV headers
    headers = ["timestamp", "cpu_ram_gb"]
    for i in range(device_count):
        headers.append(f"gpu_{i}_vram_gb")

    # Track Peak Usage for Summary
    peak_cpu = 0.0
    peak_gpu = 0.0

    # Open file in 'w' mode (overwrite) or 'a' (append)?
    # 'w' is safer for new jobs.
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        f.flush()

        try:
            while running:
                # Check if monitored process is still alive
                if not psutil.pid_exists(args.pid):
                    print(f"[Logger] Target PID {args.pid} is gone. Exiting.")
                    break

                # --- MEASURE ---
                current_time = time.time()

                # CPU RAM (Tree)
                cpu_mem = get_process_tree_memory(args.pid)
                peak_cpu = max(peak_cpu, cpu_mem)

                # GPU VRAM
                gpu_mems = get_gpu_memory(device_count)
                if gpu_mems:
                    peak_gpu = max(peak_gpu, max(gpu_mems))

                # --- LOG ---
                row = [current_time, f"{cpu_mem:.3f}"]
                row.extend([f"{m:.3f}" for m in gpu_mems])

                writer.writerow(row)
                f.flush()  # CRITICAL: Ensure data is written immediately

                time.sleep(args.interval)

        except Exception as e:
            print(f"[Logger] Error during logging loop: {e}")
        finally:
            if NVML_AVAILABLE and device_count > 0:
                try:
                    nvmlShutdown()
                except Exception:
                    pass

    # 4. Summary and Plotting
    print("-" * 40)
    print("[Logger] Finished.")
    print(f"   Peak System RAM: {peak_cpu:.2f} GB")
    if device_count > 0:
        print(f"   Peak GPU VRAM:   {peak_gpu:.2f} GB")
    print("-" * 40)

    if args.plot:
        generate_plot(args.output)


if __name__ == "__main__":
    main()
