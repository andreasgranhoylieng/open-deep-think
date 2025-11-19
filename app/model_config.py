"""
Model Registry based on Nov 2025 OpenRouter availability.
"""

# The Synthesizer
JUDGE_MODEL = "google/gemini-3.0-pro" 

# The Parallel Workers
# Worker 1 & 2: Gemini 3 (High Reasoning)
WORKER_GEMINI = "google/gemini-3.0-pro"

# Worker 3 & 4: GPT 5.1 (Thinking Variant)
# Using 'thinking' variant per leaks suggesting specialized reasoning buckets
WORKER_GPT = "openai/gpt-5.1-thinking" 

# Worker 5: DeepSeek V3 (The wildcard)
WORKER_DEEPSEEK = "deepseek/deepseek-v3"

# Default configuration for the team
DEFAULT_TEAM_CONFIG = {
    "judge": JUDGE_MODEL,
    "workers": [
        WORKER_GEMINI,          # Worker 1
        WORKER_GEMINI,          # Worker 2
        WORKER_GPT,             # Worker 3
        WORKER_GPT,             # Worker 4
        WORKER_DEEPSEEK         # Worker 5
    ]
}