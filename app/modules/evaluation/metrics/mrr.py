from typing import List, Any
from app.core.interfaces.evaluation import BaseMetric

class MRRMetric(BaseMetric):
    def __init__(self, k: int = 5):
        self.k = k

    @property
    def name(self) -> str:
        return f"MRR@{self.k}"

    def calculate(self, predictions: List[List[str]], ground_truths: List[List[str]], **kwargs) -> float:
        if not predictions or not ground_truths or len(predictions) != len(ground_truths):
            return 0.0
            
        mrr = 0.0
        for i in range(len(ground_truths)):
            preds_for_q = predictions[i][:self.k]
            gt_for_q = ground_truths[i]
            
            rr = 0.0
            for rank, pred in enumerate(preds_for_q):
                is_relevant = False
                for gt in gt_for_q:
                    if gt.strip() in pred.strip() or pred.strip() in gt.strip():
                        is_relevant = True
                        break
                if is_relevant:
                    rr = 1.0 / (rank + 1)
                    break
            mrr += rr
                
        return mrr / len(ground_truths)
