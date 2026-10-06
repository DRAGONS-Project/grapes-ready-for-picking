import abc


class BaseLogger(abc.ABC):
    """Base class for experiment loggers.

    Subclasses receive a dict with namespaced keys:
        "metrics/<name>"   → logged as W&B scalars
        "metadata/<name>"  → saved as CSV files in the run directory
    """

    @abc.abstractmethod
    def log(self, log_dict: dict):
        """Log a dict of key-value pairs."""
