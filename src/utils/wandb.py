import omegaconf
from hydra.utils import instantiate


def setup_logger(config):
    """Instantiate the W&B logger from config.

    group = date subfolder (e.g. 2025-01-15)
    name  = time subfolder (e.g. 14-32-01)
    Both are extracted from the Hydra output dir set by preprocess_config().
    """
    group, name = config.exp.log_dir.split("/")[-2:]
    wandb_config = omegaconf.OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )
    return instantiate(config.logger)(
        config=wandb_config,
        dir=config.exp.log_dir,
        group=group,
        name=name,
    )
