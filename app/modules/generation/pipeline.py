from typing import AsyncGenerator
from langchain_core.messages import HumanMessage, AIMessage

from app.core.interfaces.generation import IGenerator
from app.core.schemas.generation import GenerationRequest, GenerationChunk
from app.modules.generation.memory import ConversationMemoryManager

class GenerationPipeline:
    """
    Coordinates the generation process, including memory management
    and invoking the generator service.
    """
    
    def __init__(self, generator: IGenerator, memory_manager: ConversationMemoryManager):
        self.generator = generator
        self.memory_manager = memory_manager

    async def process(self, request: GenerationRequest) -> AsyncGenerator[GenerationChunk, None]:
        """
        Process the generation request using streaming, appending to and reading from chat history.
        """
        session_id = request.session_id
        history = self.memory_manager.get_session_history(session_id)
        
        chat_history = history.messages.copy()
        
        full_answer = ""
        
        async for chunk in self.generator.generate_stream(request, chat_history=chat_history):
            if chunk.text:
                full_answer += chunk.text
            yield chunk
            
        # Update history after streaming is done
        self.memory_manager.add_messages(
            session_id, 
            [
                HumanMessage(content=request.query),
                AIMessage(content=full_answer)
            ]
        )
