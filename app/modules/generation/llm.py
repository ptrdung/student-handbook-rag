import os
from typing import AsyncGenerator, List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from google import genai
from google.genai import types

from app.core.interfaces.generation import IGenerator, ISemanticCache, ILLM
from app.core.schemas.generation import GenerationRequest, GenerationResponse, GenerationChunk
from app.core.prompts.generation import get_generation_prompt, SYSTEM_PROMPT, HUMAN_PROMPT
from app.core.interfaces.base import IEmbedder

class LLMService(IGenerator, ILLM):
    """
    LLM Generation service utilizing NVIDIA's AI Foundation API.
    Uses 'google/gemma-3-27b-it' via OpenAI-compatible endpoint.
    """
    
    def __init__(self, semantic_cache: ISemanticCache = None, embedder: IEmbedder = None, api_key: str = None):
        self.cache = semantic_cache
        self.embedder = embedder
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY environment variable is not set")
            
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            base_url=os.getenv("LLM_URL"),
            model=os.getenv("LLM_MODEL"),
            temperature=0.2, # Low temp for factual answers
            max_tokens=1024
        )
        self.prompt_template = get_generation_prompt()
        self.chain = self.prompt_template | self.llm | StrOutputParser()
        
    def _format_context(self, request: GenerationRequest) -> str:
        """Format the retrieved documents into a single context string."""
        if not request.context:
            return ""
            
        formatted_parts = []
        for i, item in enumerate(request.context):
            formatted_parts.append(f"[Document {i+1}]:\n{item.content}")
        return "\n\n".join(formatted_parts)

    def generate(self, request: GenerationRequest, chat_history: List[Any] = None) -> GenerationResponse:
        """Generate a complete answer synchronously, with manual caching."""
        import logging
        logger = logging.getLogger(__name__)
        query_text = request.query
        
        # 1. Prepare for Generation
        context_str = self._format_context(request)
        
        answer = self.chain.invoke({
            "context_str": context_str,
            "query": request.query,
            "chat_history": chat_history or []
        })
        
        # 2. Lưu lại vào Redis (Semantic Cache)
        if query_text and answer and self.cache and self.embedder:
            query_vector = self.embedder.embed_text(query_text)
            self.cache.set(query_text, query_vector, answer)
            logger.info("💾 Đã lưu kết quả vào Semantic Cache.")
        
        return GenerationResponse(
            answer=answer,
            session_id=request.session_id
        )

    # Xóa decorator đi vì không tương thích với Generator
    async def generate_stream(self, request: GenerationRequest, chat_history: List[Any] = None) -> AsyncGenerator[GenerationChunk, None]:
        """Generate an answer and stream it back chunk by chunk, with manual caching."""
        import logging
        logger = logging.getLogger(__name__)
        query_text = request.query
        
        # 1. Gọi LLM thật để nhả Stream
        context_str = self._format_context(request)
        full_answer = "" # Chuẩn bị cái giỏ hứng chữ
        
        async for chunk in self.chain.astream({
            "context_str": context_str,
            "query": request.query,
            "chat_history": chat_history or []
        }):
            full_answer += chunk # Cộng dồn từng chữ vào giỏ
            yield GenerationChunk(
                text=chunk,
                is_last=False
            )
            
        # 4. Khi LLM nói xong, lấy giỏ chữ ném vào Redis
        if query_text and full_answer and self.cache and self.embedder:
            query_vector = self.embedder.embed_text(query_text)
            self.cache.set(query_text, query_vector, full_answer)
            logger.info("💾 Đã lưu chuỗi Stream hoàn chỉnh vào Redis Semantic Cache.")
            
        # 5. Yield tín hiệu kết thúc
        yield GenerationChunk(
            text="",
            is_last=True,
        )

    async def generate_response(self, prompt: str) -> str:
        """
        Generate a raw string response from the LLM.
        Directly uses the underlying ChatOpenAI model with a simple string parser.
        """
        from langchain_core.output_parsers import StrOutputParser

        chain = self.llm | StrOutputParser()
        return await chain.ainvoke(prompt)


class GoogleLLMService(IGenerator, ILLM):
    """
    LLM Generation service using Google AI Studio API.
    Uses 'gemma-4-31b-it' via the google-genai library.
    """

    def __init__(self, semantic_cache: ISemanticCache = None, embedder: IEmbedder = None, api_key: str = None):
        self.cache = semantic_cache
        self.embedder = embedder
        self.api_key = api_key or os.getenv("GOOGLE_AI_STUDIO_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_AI_STUDIO_API_KEY environment variable is not set")

        self.model_name = os.getenv("GOOGLE_LLM_MODEL", "gemma-4-31b-it")
        self.client = genai.Client(api_key=self.api_key)
        self.generation_config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=1024,
            temperature=0.2,
        )

    def _format_context(self, request: GenerationRequest) -> str:
        if not request.context:
            return ""
        formatted_parts = []
        for i, item in enumerate(request.context):
            formatted_parts.append(f"[Document {i+1}]:\n{item.content}")
        return "\n\n".join(formatted_parts)

    def _build_contents(self, request: GenerationRequest, chat_history: List[Any] = None) -> List[types.Content]:
        contents = []
        if chat_history:
            for msg in chat_history:
                if isinstance(msg, HumanMessage):
                    contents.append(types.Content(role="user", parts=[types.Part(text=msg.content)]))
                elif isinstance(msg, AIMessage):
                    contents.append(types.Content(role="model", parts=[types.Part(text=msg.content)]))

        human_prompt = HUMAN_PROMPT.format(
            context_str=self._format_context(request),
            query=request.query,
        )
        contents.append(types.Content(role="user", parts=[types.Part(text=human_prompt)]))
        return contents

    def generate(self, request: GenerationRequest, chat_history: List[Any] = None) -> GenerationResponse:
        import logging
        logger = logging.getLogger(__name__)

        contents = self._build_contents(request, chat_history)
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=self.generation_config,
        )
        answer = response.text

        if request.query and answer and self.cache and self.embedder:
            query_vector = self.embedder.embed_text(request.query)
            self.cache.set(request.query, query_vector, answer)
            logger.info("💾 Đã lưu kết quả vào Semantic Cache.")

        return GenerationResponse(answer=answer, session_id=request.session_id)

    async def generate_stream(self, request: GenerationRequest, chat_history: List[Any] = None) -> AsyncGenerator[GenerationChunk, None]:
        import logging
        logger = logging.getLogger(__name__)

        contents = self._build_contents(request, chat_history)
        full_answer = ""

        async for chunk in await self.client.aio.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=self.generation_config,
        ):
            if chunk.text:
                full_answer += chunk.text
                yield GenerationChunk(text=chunk.text, is_last=False)

        if request.query and full_answer and self.cache and self.embedder:
            query_vector = self.embedder.embed_text(request.query)
            self.cache.set(request.query, query_vector, full_answer)
            logger.info("💾 Đã lưu chuỗi Stream hoàn chỉnh vào Redis Semantic Cache.")

        yield GenerationChunk(text="", is_last=True)

    async def generate_response(self, prompt: str) -> str:
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text
