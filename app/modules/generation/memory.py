from typing import Dict, List
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import BaseMessage

class ConversationMemoryManager:
    """
    Manages conversation history for multi-turn interactions.
    Currently uses in-memory storage, can be extended to use Redis or DB.
    """
    
    def __init__(self):
        # In-memory store for session histories
        self.store: Dict[str, BaseChatMessageHistory] = {}
        
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Retrieve the chat history for a given session ID.
        If it doesn't exist, create a new one.
        """
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
        
    def add_messages(self, session_id: str, messages: List[BaseMessage]):
        """
        Add a list of messages to the session history.
        """
        history = self.get_session_history(session_id)
        history.add_messages(messages)
        
    def clear_history(self, session_id: str):
        """
        Clear the history for a specific session.
        """
        if session_id in self.store:
            self.store[session_id].clear()
            del self.store[session_id]
