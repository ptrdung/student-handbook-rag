import pytest
from langchain_core.messages import HumanMessage, AIMessage
from app.modules.generation.memory import ConversationMemoryManager

def test_memory_manager_get_session_history():
    manager = ConversationMemoryManager()
    
    # Getting history for a new session should create it
    history1 = manager.get_session_history("session_1")
    assert len(history1.messages) == 0
    
    # Adding a message directly to history
    history1.add_message(HumanMessage(content="Hello"))
    
    # Retrieving it again should return the same instance
    history2 = manager.get_session_history("session_1")
    assert len(history2.messages) == 1
    assert history2.messages[0].content == "Hello"
    assert history1 is history2
    
def test_memory_manager_add_messages():
    manager = ConversationMemoryManager()
    
    messages = [
        HumanMessage(content="Question 1"),
        AIMessage(content="Answer 1")
    ]
    
    manager.add_messages("session_2", messages)
    
    history = manager.get_session_history("session_2")
    assert len(history.messages) == 2
    assert history.messages[0].content == "Question 1"
    assert history.messages[1].content == "Answer 1"

def test_memory_manager_clear_history():
    manager = ConversationMemoryManager()
    
    manager.add_messages("session_3", [HumanMessage(content="Test")])
    assert len(manager.get_session_history("session_3").messages) == 1
    
    manager.clear_history("session_3")
    
    # Should create a fresh history after clearing
    history_after_clear = manager.get_session_history("session_3")
    assert len(history_after_clear.messages) == 0
