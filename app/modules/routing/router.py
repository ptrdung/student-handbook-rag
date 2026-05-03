from app.core.interfaces.routing import IQueryRouter
from app.core.interfaces.generation import ILLM
from app.core.schemas.routing import QueryType, RoutingResult
from app.core.prompts.routing import get_routing_prompt
import logging

logger = logging.getLogger(__name__)

class LLMQueryRouter(IQueryRouter):
    """
    Router that uses an LLM to classify user queries.
    """
    
    def __init__(self, llm: ILLM):
        self.llm = llm
        self.system_prompt = get_routing_prompt()

    async def route(self, query: str) -> RoutingResult:
        """
        Route a user query to the appropriate category using an LLM.
        """
        if not query or not query.strip():
            return RoutingResult(query_type=QueryType.SMALL_TALK, reasoning="Empty query")

        prompt = f"{self.system_prompt}\n\nUser Input: {query}\n\nClassification:"
        
        try:
            response = await self.llm.generate_response(prompt)
            return self._parse_response(response)
        except Exception as e:
            logger.error(f"Error during query routing: {e}")
            return RoutingResult(query_type=QueryType.SMALL_TALK, reasoning=f"Error: {str(e)}")

    def _parse_response(self, response: str) -> RoutingResult:
        """Parse the raw LLM response into a RoutingResult."""
        response = response.strip()
        
        for q_type in QueryType:
            if response.startswith(q_type.value):
                reasoning = response[len(q_type.value):].lstrip(": ").strip()
                return RoutingResult(query_type=q_type, reasoning=reasoning)
        
        # Fallback logic if no category matches
        return RoutingResult(query_type=QueryType.SMALL_TALK, reasoning=f"Ambiguous response: {response}")
