from typing import List, Any
import math
from app.core.interfaces.evaluation import BaseMetric

class NDCGMetric(BaseMetric):
    def __init__(self, k: int = 5):
        self.k = k

    @property
    def name(self) -> str:
        return f"nDCG@{self.k}"

    def calculate(self, predictions: List[List[str]], ground_truths: List[List[str]], **kwargs) -> float:
        if not predictions or not ground_truths or len(predictions) != len(ground_truths):
            return 0.0
            
        ndcg = 0.0
        for i in range(len(ground_truths)):
            preds_for_q = predictions[i][:self.k]
            gt_for_q = ground_truths[i]
            
            dcg = 0.0
            
            # calculate dcg
            for rank, pred in enumerate(preds_for_q):
                is_relevant = False
                for gt in gt_for_q:
                    if gt.strip() in pred.strip() or pred.strip() in gt.strip():
                        is_relevant = True
                        break
                
                if is_relevant:
                    rel_i = 1.0
                    dcg += rel_i / math.log2((rank + 1) + 1)
            
            # calculate idcg (ideal dcg where all relevant chunks come first)
            num_relevant = len(gt_for_q)
            idcg = 0.0
            for rank in range(min(num_relevant, self.k)):
                idcg += 1.0 / math.log2((rank + 1) + 1)
                
            if idcg > 0:
                ndcg += dcg / idcg
                
        return ndcg / len(ground_truths)
