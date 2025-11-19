import os
from typing import List, Tuple
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import LLM
from app.model_config import DEFAULT_TEAM_CONFIG

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
def _get_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is required to call OpenRouter; set the env var before starting the app."
        )
    return api_key

def get_openrouter_llm(model_name: str, temperature: float, role: str) -> OpenAI:
    """
    Factory for OpenRouter LLMs.
    Args:
        role: 'judge' or 'worker' (used for logging/headers)
    """
    return OpenAI(
        model=model_name,
        api_base=OPENROUTER_API_BASE,
        api_key=_get_api_key(),
        temperature=temperature,
        # GPT-5.1 and Gemini 3 have massive contexts, safe to set high
        max_tokens=8192, 
        default_headers={
            "HTTP-Referer": "https://github.com/your-repo/deep-think-2025",
            "X-Title": f"DeepThink {role.capitalize()}",
        }
    )

def build_specific_team() -> Tuple[List[LLM], LLM]:
    """
    Constructs the exact 5-worker + 1-judge team requested.
    """
    config = DEFAULT_TEAM_CONFIG
    
    # 1. The Judge: Gemini 3 (Low temp for precise synthesis)
    judge = get_openrouter_llm(config["judge"], temperature=0.1, role="judge")
    
    # 2. The Workers (Higher temp for diverse reasoning)
    workers = []
    
    # Add workers dynamically based on list
    for idx, model_id in enumerate(config["workers"]):
        # We use slightly higher temp for DeepSeek to get "out of box" ideas
        temp = 0.85 if "deepseek" in model_id else 0.7
        llm = get_openrouter_llm(model_id, temperature=temp, role=f"worker_{idx}")
        workers.append(llm)
        
    return workers, judge