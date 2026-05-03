from typing import List
from app.core.interfaces.evaluation import BaseEvaluator, BaseMetric, MetricTracker
from app.core.schemas.evaluation import EvaluationRecord, MetricResult, EvaluationResult
from app.core.interfaces.retrieval import ISearchEngine
from app.core.schemas.retrieval import SearchRequest

class SearchEvaluator(BaseEvaluator):
    def __init__(self, search_pipeline: ISearchEngine, metrics: List[BaseMetric], tracker: MetricTracker = None):
        super().__init__(metrics)
        self.search_pipeline = search_pipeline
        self.tracker = tracker

    def evaluate(self, dataset: List[EvaluationRecord], use_rerank: bool = False, limit_search: int = 10, limit_rerank: int = 5, **kwargs) -> EvaluationResult:
        predictions = []
        ground_truths = []
        
        for record in dataset:
            request = SearchRequest(
                query=record.question,
                limit_search=limit_search,
                use_rerank=use_rerank,
                limit_rerank=limit_rerank
            )
            
            results = self.search_pipeline.search(request)
            retrieved_chunks = [res.content for res in results]
            
            predictions.append(retrieved_chunks)
            ground_truths.append(record.contexts)
            
        metric_results = []
        summary = {}
        
        for metric in self.metrics:
            score = metric.calculate(predictions, ground_truths)
            metric_results.append(MetricResult(name=metric.name, value=score))
            summary[metric.name] = score
            
            if self.tracker:
                self.tracker.log_metric(metric.name, score)
                
        if self.tracker:
            self.tracker.finish()
            
        return EvaluationResult(metrics=metric_results, summary=summary)
