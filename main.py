import chainlit as cl
from app.engine import DeepThinkWorkflow
from app.ui_utils import build_specific_team

@cl.on_chat_start
async def start():
    # Load the 5 Workers + 1 Judge configuration
    workers, judge = build_specific_team()
    
    workflow = DeepThinkWorkflow(workers=workers, judge=judge, timeout=180)
    cl.user_session.set("workflow", workflow)
    
    # Display the team roster
    roster = "\n".join([f" - 🤖 {w.metadata.model_name}" for w in workers])
    await cl.Message(
        content=f"""
# 🧠 DeepThink 2025 Ready
**Judge:** {judge.metadata.model_name}
**Worker Matrix (5 Parallel Threads):**
{roster}
"""
    ).send()

@cl.on_message
async def main(message: cl.Message):
    workflow: DeepThinkWorkflow = cl.user_session.get("workflow")
    
    msg_final = cl.Message(content="")
    
    async with cl.Step(name="Thinking Matrix (5 Models)") as step:
        step.input = message.content
        
        # Execute Workflow
        result = await workflow.run(query=message.content)
        
        # Display Parallel Thoughts inside the collapsible step
        step.elements = [
            cl.Text(name="Parallel Reasoning", content=result["thoughts"], display="inline")
        ]
        step.output = "Thoughts Aggregated. Synthesizing with Gemini 3..."
        
        # Stream Final Answer
        await msg_final.send()
        final_content = ""
        async for chunk in result["response_gen"]:
            delta = getattr(chunk, "delta", None)
            if not delta:
                message_content = getattr(chunk.message, "content", "")
                delta = message_content[len(final_content):]
            if delta:
                await msg_final.stream_token(delta)
                final_content += delta
        await msg_final.update(content=final_content)