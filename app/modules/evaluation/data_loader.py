import json
from typing import List
from app.core.schemas.evaluation import EvaluationRecord

class JSONDataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[EvaluationRecord]:
        records = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                # Expecting format: { "question": str, "contexts": List[str], "answer": str }
                record = EvaluationRecord(
                    question=item.get("question", ""),
                    contexts=item.get("contexts", []),
                    answer=item.get("answer", "")
                )
                records.append(record)
        return records
