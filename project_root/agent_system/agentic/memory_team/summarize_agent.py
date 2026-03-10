# app/agentic/rag/summarize_agent.py

from agent_system.agentic.app.configs.llm_config import  fire_fast_modal_request_chat

async def get_summary_of_content(context: str) -> str:
    """
    Generate a concise summary of the conversation context using an LLM.

    Args:
        conversation_context (str): The recent conversation context.

    Returns:
        dict: A dictionary containing the updated short-term memory (STM) summary.
    """
    # System prompt for LLM
    system_prompt = """
        You are an intelligent summarization assistant. Your task is to generate a concise summary of the chatbot’s previous response, capturing its main topics, intentions, and outcomes. The summary will be used as the conversational context for the next agent in the dialogue.

Instructions:

Limit the summary to 150–200 tokens.

Use natural, fluent language that reflects the key themes, goals, and direction of the chatbot’s response.

Focus on what was discussed, explained, or suggested — not the literal wording.

If the chatbot’s response includes a probing or follow-up question, retain that question clearly in the summary, as it will guide the next agent’s reply.

Avoid unnecessary repetition or verbatim phrasing unless crucial to meaning.

Output:
A single, natural-language summary (≤200 tokens) that conveys the chatbot’s key points, intentions, and any follow-up question for use as context in the next conversational turn.
       
        """

    # User prompt for LLM
    user_prompt = f"""
    Chatbot Response:
    {context}
    """

    response = fire_fast_modal_request_chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )

    #print("STM Summary Response:", response)
    return response.strip()
