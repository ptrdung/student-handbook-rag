import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from app.modules.generation.nvidia_gemma import NvidiaGemmaService
from app.core.schemas.generation import GenerationRequest
from app.core.schemas.retrieval import RerankResult

@pytest.fixture
def mock_env_vars(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test_api_key")

@pytest.fixture
def mock_chat_openai():
    with patch('app.modules.generation.nvidia_gemma.ChatOpenAI') as mock:
        yield mock

@pytest.fixture
def generation_service(mock_env_vars, mock_chat_openai):
    return NvidiaGemmaService()

@pytest.mark.asyncio
async def test_nvidia_gemma_service_generate(generation_service, mock_chat_openai):
    # Setup mock
    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = "Đây là câu trả lời mock."
    generation_service.chain = mock_chain
    
    request = GenerationRequest(
        query="Thời gian nghỉ học tạm thời tối đa là bao lâu?",
        context=[
            RerankResult(
                text="Thời gian nghỉ học tạm thời tối đa là 4 học kỳ.",
                score=0.9,
                metadata={"source_id": "doc_123", "page_number": 15}
            )
        ]
    )
    
    response = await generation_service.generate(request)
    
    assert response.answer == "Đây là câu trả lời mock."
    assert len(response.citations) == 1
    assert response.citations[0].source_id == "doc_123"
    assert response.citations[0].page_number == 15
    assert response.session_id == "default_session"
    assert mock_chain.ainvoke.called

@pytest.mark.asyncio
async def test_nvidia_gemma_service_generate_empty_context(generation_service, mock_chat_openai):
    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = "Rất tiếc, tôi không tìm thấy thông tin này trong Sổ tay Sinh viên."
    generation_service.chain = mock_chain
    
    request = GenerationRequest(query="Câu hỏi không có trong sổ tay")
    
    response = await generation_service.generate(request)
    
    assert response.answer == "Rất tiếc, tôi không tìm thấy thông tin này trong Sổ tay Sinh viên."
    assert len(response.citations) == 0

@pytest.mark.asyncio
async def test_nvidia_gemma_service_generate_stream(generation_service, mock_chat_openai):
    # Setup mock chain astream
    mock_chain = AsyncMock()
    
    async def mock_astream(*args, **kwargs):
        yield "Chunk 1 "
        yield "Chunk 2"
        
    mock_chain.astream = mock_astream
    generation_service.chain = mock_chain
    
    request = GenerationRequest(
        query="Test stream",
        context=[
            RerankResult(
                text="Some text",
                score=0.9,
                metadata={"source_id": "doc_456"}
            )
        ]
    )
    
    chunks = []
    async for chunk in generation_service.generate_stream(request):
        chunks.append(chunk)
        
    assert len(chunks) == 3 # 2 text chunks + 1 final citation chunk
    assert chunks[0].text == "Chunk 1 "
    assert chunks[0].is_last is False
    assert chunks[1].text == "Chunk 2"
    assert chunks[1].is_last is False
    assert chunks[2].text == ""
    assert chunks[2].is_last is True
    assert len(chunks[2].citations) == 1
    assert chunks[2].citations[0].source_id == "doc_456"
