from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv
from openai.types.responses import ResponseInputItemParam
load_dotenv()


def generate_response(openai_key: str, user_message: str, context: str,
                      conversation_history: List[ResponseInputItemParam], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""

    # DONE: Define system prompt
    system_prompt = """You are a NASA expert. Answer questions using ONLY the provided context.
- If context is provided, cite the specific mission/source from the document metadata
- If context is empty or doesn't contain relevant information, say "I don't have information about that in my documents."
- Never make up sources"""
    # DONE: Set context in messages
    user_prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {user_message}

If you found relevant information, format as:
Answer: [your answer] (Source: [mission name from context])

If no relevant context was provided, say so clearly."""

    # DONE: Add chat history
    messages = conversation_history
    messages.append(
        {"role": "user", "content": user_prompt},
    )

    # DONE: Creaet OpenAI Client
    client = OpenAI(api_key=openai_key)

    # DONE: Send request to OpenAI
    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=messages,
        max_output_tokens=200,
        temperature=0.1
    )

    # DONE: Return response
    return response.output_text
