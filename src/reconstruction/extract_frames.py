"""Extract frames from a video at a fixed sampling rate."""

import argparse
from pathlib import Path

import cv2

from config import load_config


def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps: float,
    max_dim: int | None = None,
    max_frames: int | None = None,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    step = max(1, round(source_fps / fps))
    if max_frames and total_frames > 0 and total_frames // step > max_frames:
        step = max(1, total_frames // max_frames)

    frame_idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            if max_dim:
                h, w = frame.shape[:2]
                scale = max_dim / max(h, w)
                if scale < 1:
                    frame = cv2.resize(
                        frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
                    )
            out_path = output_dir / f"frame_{saved:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
            if max_frames and saved >= max_frames:
                break
        frame_idx += 1

    cap.release()
    return saved


def main():
    config = load_config()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Input video file")
    parser.add_argument("output_dir", type=Path, help="Directory to write extracted frames")
    parser.add_argument(
        "--fps", type=float, default=config.frames.fps, help="Target sampling rate in frames per second"
    )
    parser.add_argument(
        "--max-dim", type=int, default=config.frames.max_dim, help="Resize longest image side to this many pixels"
    )
    parser.add_argument(
        "--max-frames", type=int, default=config.frames.max_frames, help="Cap on the number of extracted frames"
    )
    args = parser.parse_args()

    n = extract_frames(args.video, args.output_dir, args.fps, args.max_dim, args.max_frames)
    print(f"Extracted {n} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
