import logging

import utils
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


def run(config: DictConfig):
    # --- Setup ---
    utils.hydra.preprocess_config(config)
    logger = utils.wandb.setup_logger(config)
    device = utils.training.setup_device(config)
    utils.training.set_seed(config.exp.seed)

    log.info(f"Running on device: {device}")

    # --- Components ---
    # Instantiate model, dataset, and metrics from config.
    # To customize: replace model/dataset/metric configs or add your own subclasses.
    model = instantiate(config.model).to(device)

    dataset = instantiate(config.dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=config.dataset.dataloader.batch_size,
        num_workers=config.dataset.dataloader.num_workers,
        shuffle=config.dataset.dataloader.shuffle,
        pin_memory=config.dataset.dataloader.pin_memory,
    )

    metrics = {name: instantiate(cfg) for name, cfg in config.metric.items()}

    # --- Main Loop ---
    # Replace this section with your task-specific logic:
    #
    # Classic ML:
    #   optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #   for epoch in range(config.exp.epochs):
    #       for batch in dataloader:
    #           loss = model(batch)
    #           optimizer.zero_grad(); loss.backward(); optimizer.step()
    #           logger.log({"metrics/loss": loss.item()})
    #
    # RL:
    #   env = gymnasium.make(config.env.id)
    #   for step in range(config.exp.total_steps):
    #       rollout = collect_rollout(model, env)
    #       loss = update_policy(model, rollout, optimizer)
    #       logger.log({"metrics/reward": rollout.mean_reward})
    #
    # XAI:
    #   for batch in dataloader:
    #       attributions = analyze(model, batch, method=config.exp.method)
    #       visualize(attributions, logger)

    for batch in dataloader:
        pass  # replace with forward pass / analysis / rollout

    # --- Metrics ---
    for name, metric in metrics.items():
        result = metric.compute_and_log()
        log.info(f"{name}: {result}")
        logger.log({f"metrics/{name}": result})
