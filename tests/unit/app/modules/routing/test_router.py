import pytest
from unittest.mock import AsyncMock, MagicMock
from app.modules.routing.router import LLMQueryRouter
from app.core.schemas.routing import QueryType, RoutingResult
from app.core.interfaces.generation import ILLM

@pytest.mark.asyncio
async def test_llm_router_academic_policy():
    # Mock LLM response for academic policy
    mock_llm = MagicMock(spec=ILLM)
    mock_llm.generate_response = AsyncMock(return_value="ACADEMIC_POLICY: The user is asking about tuition fees.")
    
    router = LLMQueryRouter(llm=mock_llm)
    result = await router.route("What are the tuition fees for 2024?")
    
    assert result.query_type == QueryType.ACADEMIC_POLICY
    assert result.reasoning is not None
    assert "tuition fees" in result.reasoning.lower()

@pytest.mark.asyncio
async def test_llm_router_greeting():
    mock_llm = MagicMock(spec=ILLM)
    mock_llm.generate_response = AsyncMock(return_value="GREETING_CLOSING: User said hello.")
    
    router = LLMQueryRouter(llm=mock_llm)
    result = await router.route("Hi there!")
    
    assert result.query_type == QueryType.GREETING_CLOSING

@pytest.mark.asyncio
async def test_llm_router_small_talk():
    mock_llm = MagicMock(spec=ILLM)
    mock_llm.generate_response = AsyncMock(return_value="SMALL_TALK: User is talking about weather.")
    
    router = LLMQueryRouter(llm=mock_llm)
    result = await router.route("How is the weather today?")
    
    assert result.query_type == QueryType.SMALL_TALK

@pytest.mark.asyncio
async def test_llm_router_fallback():
    mock_llm = MagicMock(spec=ILLM)
    # LLM returns garbage
    mock_llm.generate_response = AsyncMock(return_value="I don't know what you mean.")
    
    router = LLMQueryRouter(llm=mock_llm)
    result = await router.route("Random string 123")
    
    # Should default to SMALL_TALK
    assert result.query_type == QueryType.SMALL_TALK

@pytest.mark.asyncio
async def test_llm_router_out_of_scope():
    mock_llm = MagicMock(spec=ILLM)
    mock_llm.generate_response = AsyncMock(return_value="OUT_OF_SCOPE: Politics is not related to university.")
    
    router = LLMQueryRouter(llm=mock_llm)
    result = await router.route("Who is the president?")
    
    assert result.query_type == QueryType.OUT_OF_SCOPE
