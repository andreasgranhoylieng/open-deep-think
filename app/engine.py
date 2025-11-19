import asyncio
from typing import List
from llama_index.core.workflow import (
    Workflow, StartEvent, StopEvent, step, Context, Event
)
from llama_index.core.llms import LLM, ChatMessage

# Specialized prompts for 2025 Reasoning Models
THINKER_PROMPT = """
You are an advanced reasoning agent (Model: {model_name}).
You are 1 of 5 parallel workers solving a hard problem.

TASK:
1. Do NOT give a simple answer. 
2. Perform a "Deep Dive" analysis.
3. If you are GPT-5.1, focus on structural logic and documentation.
4. If you are Gemini 3, focus on mathematics and code.
5. If you are DeepSeek, look for edge cases and code-based proofs.

Output your internal monologue clearly marked with markdown headers.
Question: {query}
"""

JUDGE_PROMPT_GEMINI_3 = """
You are Gemini 3.0 Pro, the Master Synthesizer.
You have received 5 distinct thought paths from top-tier AI models, including yourself (previous instances) and GPT-5.1.

YOUR GOAL:
Determine the absolute truth by cross-referencing these 5 streams.
1. Identify where GPT-5.1 and DeepSeek V3 disagree, if anywhere.
2. If calculations differ, verify them step-by-step yourself.
3. Synthesize the final answer in a clean, authoritative format.

---
INPUT DATA (Parallel Thoughts):
{thoughts}
"""

class AnalysisEvent(Event):
    thoughts: str

class DeepThinkWorkflow(Workflow):
    def __init__(self, workers: List[LLM], judge: LLM, timeout: int = 600, **kwargs):
        super().__init__(timeout=timeout, **kwargs)
        self.workers = workers
        self.judge = judge

    @step
    async def parallel_think(self, ctx: Context, ev: StartEvent) -> AnalysisEvent:
        query = ev.get("query")
        await ctx.set("query", query)

        async def run_worker(worker: LLM, idx: int):
            model_name = worker.metadata.model_name
            # Personalized prompt injection
            prompt = THINKER_PROMPT.format(model_name=model_name, query=query)
            try:
                res = await worker.achat([ChatMessage(role="user", content=prompt)])
                return f"## 🧠 Worker {idx+1}: {model_name}\n{res.message.content}\n\n---\n"
            except Exception as e:
                return f"## ❌ Worker {idx+1} ({model_name}) Failed: {str(e)}\n"

        # Spawn all 5 workers
        tasks = [run_worker(w, i) for i, w in enumerate(self.workers)]
        results = await asyncio.gather(*tasks)
        
        return AnalysisEvent(thoughts="\n".join(results))

    @step
    async def synthesize(self, ctx: Context, ev: AnalysisEvent) -> StopEvent:
        query = await ctx.get("query")
        prompt = JUDGE_PROMPT_GEMINI_3.format(thoughts=ev.thoughts)
        
        # Stream the Gemini 3 Judge response
        response_gen = await self.judge.astream_chat(
            [ChatMessage(role="user", content=prompt)]
        )
        
        return StopEvent(result={
            "response_gen": response_gen,
            "thoughts": ev.thoughts
        })