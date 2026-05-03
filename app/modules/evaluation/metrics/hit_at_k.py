from typing import List, Any
from app.core.interfaces.evaluation import BaseMetric

class HitAtKMetric(BaseMetric):
    def __init__(self, k: int = 5):
        self.k = k

    @property
    def name(self) -> str:
        return f"Hit@{self.k}"

    def calculate(self, predictions: List[List[str]], ground_truths: List[List[str]], **kwargs) -> float:
        if not predictions or not ground_truths or len(predictions) != len(ground_truths):
            return 0.0
            
        hits = 0
        for i in range(len(ground_truths)):
            preds_for_q = predictions[i][:self.k]
            gt_for_q = ground_truths[i]
            
            # Hit occurs if ANY retrieved chunk in top-k is relevant (matches ANY gold_span)
            hit = False
            for pred in preds_for_q:
                for gt in gt_for_q:
                    if gt.strip() in pred.strip() or pred.strip() in gt.strip():
                        hit = True
                        break
                if hit:
                    break
                    
            if hit:
                hits += 1
                
        return hits / len(ground_truths)
