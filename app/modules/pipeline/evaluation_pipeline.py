import logging
from typing import Dict, Any
from app.core.interfaces.evaluation import BaseEvaluator
from app.modules.evaluation.data_loader import JSONDataLoader
from app.core.schemas.evaluation import EvaluationResult

logger = logging.getLogger(__name__)

class EvaluationPipeline:
    """
    High-level orchestrator for running quantitative evaluations on the system.
    """
    def __init__(self, evaluators: Dict[str, BaseEvaluator]):
        """
        Initializes the Evaluation Pipeline with a registry of evaluators.
        
        Args:
            evaluators: A dictionary mapping evaluator names to BaseEvaluator instances.
        """
        self.evaluators = evaluators

    def run_evaluation(self, evaluator_name: str, data_path: str, **kwargs) -> EvaluationResult:
        """
        Runs an evaluation using the specified evaluator.
        
        Args:
            evaluator_name: The registered name of the evaluator to run.
            data_path: Path to the JSON evaluation dataset.
            **kwargs: Additional arguments to pass to the evaluator (e.g., use_rerank, limit_search).
            
        Returns:
            EvaluationResult containing all computed metrics.
        """
        if evaluator_name not in self.evaluators:
            raise ValueError(f"Evaluator '{evaluator_name}' not found. Available evaluators: {list(self.evaluators.keys())}")
            
        logger.info(f"Loading evaluation dataset from {data_path}...")
        loader = JSONDataLoader(data_path)
        dataset = loader.load()
        logger.info(f"Loaded {len(dataset)} evaluation records.")
        
        logger.info(f"Running Evaluation with '{evaluator_name}' (kwargs={kwargs})...")
        evaluator = self.evaluators[evaluator_name]
        result = evaluator.evaluate(dataset=dataset, **kwargs)
        
        logger.info("Evaluation completed successfully.")
        return result
