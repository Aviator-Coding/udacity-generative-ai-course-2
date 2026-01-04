from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""

    # TODO: Define system prompt
    # TODO: Set context in messages
    system_prompt=""" You are NASA expert your goal is to take the user's question and the context from your RAG client and generate a helpful, human-readable answer

USERS CONTEXT:
{context}
"""

    # TODO: Add chat history
    messages = conversation_history
    messages.append(
         {"role": "user", "content": user_message},
    )
    # TODO: Creaet OpenAI Client
    client = OpenAI()
    # TODO: Send request to OpenAI
    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=messages
    )

    # TODO: Return response
    return response.output_text
