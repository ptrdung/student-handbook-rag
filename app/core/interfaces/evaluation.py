from abc import ABC, abstractmethod
from typing import List, Dict, Any
from app.core.schemas.evaluation import EvaluationRecord, MetricResult, EvaluationResult

class BaseMetric(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the metric."""
        pass

    @abstractmethod
    def calculate(self, predictions: List[Any], ground_truths: List[Any], **kwargs) -> float:
        """Calculate the metric value based on predictions and ground truths."""
        pass

class BaseEvaluator(ABC):
    def __init__(self, metrics: List[BaseMetric]):
        self.metrics = metrics

    @abstractmethod
    def evaluate(self, dataset: List[EvaluationRecord], **kwargs) -> EvaluationResult:
        """Evaluate the dataset and return the results."""
        pass

class MetricTracker(ABC):
    @abstractmethod
    def log_metric(self, name: str, value: float, step: int = None) -> None:
        """Log a single metric."""
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: int = None) -> None:
        """Log multiple metrics."""
        pass

    @abstractmethod
    def finish(self) -> None:
        """Finish tracking and finalize any pending operations."""
        pass
