import time
import statistics
import logging
from unittest.mock import MagicMock
from dotenv import load_dotenv
from app.core.services.cache import RedisSemanticCache
from app.core.services.redis_service import RedisService
from app.core.services.embedder import BkaiVietnameseEmbedder

# Load environment variables
load_dotenv()

# Setup logging to see cache events
logging.basicConfig(level=logging.INFO)

def measure_latency():
    service = RedisService()
    if not service.is_healthy():
        print("❌ Redis is not available. Skipping latency measurement.")
        return

    embedder = BkaiVietnameseEmbedder()
    cache = RedisSemanticCache(service, threshold=0.9)
    if cache.cache:
        cache.cache.clear()

    query = "Quy định về việc đăng ký môn học muộn?"
    vector = embedder.embed_text(query)
    response = "Sinh viên có thể đăng ký muộn trong vòng 1 tuần đầu của học kỳ..."

    # 1. Warm up & Set Cache
    cache.set(query, vector, response)
    
    # 2. Measure Hit Latency (10 runs)
    hit_latencies = []
    for _ in range(10):
        start_time = time.perf_counter()
        # Simulation of what the decorator does:
        # v = embedder.embed_text(query) # We already have it, but real flow embeds again
        _ = cache.get(vector)
        end_time = time.perf_counter()
        hit_latencies.append((end_time - start_time) * 1000)

    # 3. Measure Miss Latency (Simulation of LLM call ~2s)
    # Average LLM response time is around 2000-5000ms
    llm_simulated_delay = 3000 
    
    avg_hit = statistics.mean(hit_latencies)
    p95_hit = statistics.quantiles(hit_latencies, n=20)[18] if len(hit_latencies) >= 20 else max(hit_latencies)
    
    improvement = ((llm_simulated_delay - avg_hit) / llm_simulated_delay) * 100

    print("\n--- Redis Semantic Cache Latency Report ---")
    print(f"Avg Cache Hit Latency: {avg_hit:.2f} ms")
    print(f"P95 Cache Hit Latency: {p95_hit:.2f} ms")
    print(f"Simulated LLM Latency: {llm_simulated_delay:.2f} ms")
    print(f"Latency Improvement:   {improvement:.2f} %")
    print("-------------------------------------------\n")

if __name__ == "__main__":
    measure_latency()
