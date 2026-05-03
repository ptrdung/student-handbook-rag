import pytest
import tempfile
import json
import os
from app.modules.evaluation.data_loader import JSONDataLoader

def test_json_data_loader():
    test_data = [
        {
            "question": "What is AI?",
            "contexts": ["AI is artificial intelligence."],
            "answer": "Artificial Intelligence"
        },
        {
            "question": "What is ML?",
            "contexts": ["ML is machine learning."],
            "answer": "Machine Learning"
        }
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
        json.dump(test_data, tmp)
        tmp_path = tmp.name
        
    try:
        loader = JSONDataLoader(tmp_path)
        records = loader.load()
        assert len(records) == 1
        assert records[0].question == "What is AI?"
        assert records[0].contexts == ["AI is artificial intelligence."]
        assert records[0].answer == "Artificial Intelligence"
        assert records[1].question == "What is ML?"
        assert records[1].contexts == ["ML is machine learning."]
        assert records[1].answer == "Machine Learning"
    finally:
        os.remove(tmp_path)
