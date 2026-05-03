import os
import re
import base64
import requests
from google import genai
from google.genai import types
from app.core.interfaces.ingestion import IVisionAnalyzer
from app.core.prompts.ingestion import VISION_SYSTEM_PROMPT
import time

class NvidiaVisionAnalyzer(IVisionAnalyzer):
    """Implementation of vision analysis using NVIDIA Gemma-3 API."""
    
    def __init__(self, api_key: str = None, invoke_url: str = None, model_name: str = None):
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        self.invoke_url = invoke_url or os.getenv("NVIDIA_INVOKE_URL", "https://integrate.api.nvidia.com/v1/chat/completions")
        self.model_name = model_name or os.getenv("NVIDIA_VISION_MODEL", "google/gemma-3-27b-it")

    def summarize_image(self, image_bytes: bytes) -> str:
        print(f"      [*] Triggering Deep Vision Analysis ({self.model_name})...")
        try:
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json"
            }
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": VISION_SYSTEM_PROMPT},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                        ]
                    }
                ],
                "max_tokens": 1024,
                "temperature": 0.20
            }
            
            response = requests.post(self.invoke_url, headers=headers, json=payload)
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content'].strip()
                analysis = re.search(r"<analysis>(.*?)</analysis>", content, re.DOTALL)
                markdown_content = re.search(r"<markdown_content>(.*?)</markdown_content>", content, re.DOTALL)
                if analysis and markdown_content:
                    return f"\\n\\n--- AI VISION ANALYSIS (STRICT XML) ---\\n{analysis.group(1)}\\n{markdown_content.group(1)}\\n---------------------------------------\\n"
                else:
                    return f"\\n\\n--- AI VISION ANALYSIS (STRICT XML) ---\\n{content}\\n---------------------------------------\\n"
            else:
                print(f"      [!] API Error: {response.status_code} - {response.text}")
                return ""
        except Exception as e:
            print(f"      [!] LLM Error: {e}")
            return ""


class GoogleVisionAnalyzer(IVisionAnalyzer):
    """Implementation of vision analysis using Google AI Studio Gemma-4 API."""

    def __init__(self, api_key: str = None, model_name: str = None):
        self.api_key = api_key or os.getenv("GOOGLE_AI_STUDIO_API_KEY")
        self.model_name = model_name or os.getenv("GOOGLE_VISION_MODEL", "gemma-4-31b-it")

    def summarize_image(self, image_bytes: bytes) -> str:
        if not self.api_key:
            print("      [!] Error: GOOGLE_AI_STUDIO_API_KEY not found in environment.")
            return ""

        print(f"      [*] Triggering Deep Vision Analysis ({self.model_name})...")
        try:
            time.sleep(60)  # Simulate network latency
            client = genai.Client(api_key=self.api_key)
            image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
            response = client.models.generate_content(
                model=self.model_name,
                contents=[VISION_SYSTEM_PROMPT, image_part],
                config=types.GenerateContentConfig(
                    max_output_tokens=1024,
                    temperature=0.20,
                ),
            )
            content = response.text.strip()
            analysis = re.search(r"<analysis>(.*?)</analysis>", content, re.DOTALL)
            markdown_content = re.search(r"<markdown_content>(.*?)</markdown_content>", content, re.DOTALL)
            if analysis and markdown_content:
                return f"\\n\\n--- AI VISION ANALYSIS (STRICT XML) ---\\n{analysis.group(1)}\\n{markdown_content.group(1)}\\n---------------------------------------\\n"
            else:
                return f"\\n\\n--- AI VISION ANALYSIS (STRICT XML) ---\\n{content}\\n---------------------------------------\\n"
        except Exception as e:
            print(f"      [!] LLM Error: {e}")
            return ""