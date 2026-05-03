from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Optional
from app.core.schemas.generation import GenerationRequest, GenerationResponse, GenerationChunk

class IGenerator(ABC):
    """
    Interface for the LLM Generation service.
    Responsible for taking retrieved context and a query, and generating a final answer.
    """

    @abstractmethod
    def generate(self, request: GenerationRequest, chat_history: list = None) -> GenerationResponse:
        """
        Generate a complete answer in a single non-streaming response.
        
        Args:
            request (GenerationRequest): The request containing query, context, and session info.
            chat_history (list): Optional chat history messages.
            
        Returns:
            GenerationResponse: The generated answer, citations, and metadata.
        """
        pass

    @abstractmethod
    async def generate_stream(self, request: GenerationRequest, chat_history: list = None) -> AsyncGenerator[GenerationChunk, None]:
        """
        Generate an answer and stream it back chunk by chunk.
        
        Args:
            request (GenerationRequest): The request containing query, context, and session info.
            chat_history (list): Optional chat history messages.
            
        Yields:
            GenerationChunk: A chunk of the generated response text.
        """
        pass

class ILLM(ABC):
    """Generic interface for LLM calls."""
    @abstractmethod
    async def generate_response(self, prompt: str) -> str:
        """Generate a complete response for a given prompt."""
        pass

class ISemanticCache(ABC):
    @abstractmethod
    def get(self, query_vector: List[float]) -> Optional[str]:
        """Retrieve cached response based on semantic similarity of the query vector."""
        pass

    @abstractmethod
    def set(self, query_text: str, query_vector: List[float], response: str) -> None:
        """Store query and its response in the semantic cache."""
        pass
