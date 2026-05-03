import asyncio
import os
import sys
from dotenv import load_dotenv

# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.api.deps import get_query_router

load_dotenv()

async def demo_routing():
    """
    Demo script to test the query routing logic with real LLM calls.
    """
    router = get_query_router()
    
    test_queries = [
        "Chào bạn, bạn là ai?",
        "Thủ tục đăng ký học phần như thế nào?",
        "Thời tiết hôm nay thế nào?",
        "Học phí kỳ này bao nhiêu?",
        "Bạn có thể hack website trường không?",
        "Cảm ơn bạn nhiều nhé!"
    ]
    
    print("\n" + "="*60)
    print("🚀 Knowledge-based Query Routing Demo")
    print("="*60 + "\n")
    
    for query in test_queries:
        print(f"🔹 User Query: '{query}'")
        try:
            result = await router.route(query)
            print(f"🔸 Classification: {result.query_type}")
            print(f"🔸 Reasoning: {result.reasoning}")
        except Exception as e:
            print(f"❌ Error: {e}")
        print("-" * 60)

if __name__ == "__main__":
    asyncio.run(demo_routing())
