import abc


class BaseMetric(abc.ABC):
    """Base class for metrics.

    Subclass this and implement update() and compute_and_log(). Register your
    subclass in configs/metric/default.yaml via _target_: metric.<module>.<Class>.

    Convention:
        - Call update(batch, output) inside the training/eval loop to accumulate state.
        - Call compute_and_log() at the end of an epoch or experiment to finalize.
        - Return a dict of {metric_name: value} from compute_and_log() so the caller
          can log with logger.log({"metrics/<name>": value}).
    """

    @abc.abstractmethod
    def update(self, *args, **kwargs):
        """Accumulate intermediate state from one batch."""

    @abc.abstractmethod
    def compute_and_log(self) -> dict:
        """Compute final metric values from accumulated state.

        Returns:
            dict mapping metric name to scalar value.
        """
