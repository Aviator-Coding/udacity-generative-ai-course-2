from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv
from openai.types.responses import ResponseInputItemParam
load_dotenv()


def generate_response(openai_key: str, user_message: str, context: str,
                      conversation_history: List[ResponseInputItemParam], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""

    # DONE: Define system prompt
    system_prompt = """ You are NASA expert your goal is to take the user's question and use only your context from your RAG client and generate a helpful, human-readable answer.
You must always include the data source"""
    # DONE: Set context in messages
    user_prompt = f""" Based on the following context, answer the question clearly and concisely.

Context:
{context}

Question: {user_message}

Output Format:
Answer: <youre answer> (<the source>)"""

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
