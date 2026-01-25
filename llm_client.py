from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv
from openai.types.responses import ResponseInputItemParam
load_dotenv()

# System prompt defining the NASA mission expert persona
NASA_EXPERT_SYSTEM_PROMPT = """You are a NASA Mission Expert Assistant specializing in historical space missions including Apollo, Challenger, and other NASA programs.

Your role is to provide accurate, factual information about NASA missions based ONLY on the retrieved document context provided to you.

CRITICAL REQUIREMENTS:
1. ALWAYS cite your sources explicitly using the mission name and document source from the context metadata (e.g., "According to Apollo 13 technical transcripts..." or "(Source: Apollo 11 Command Module logs)").
2. If the provided context contains relevant information, answer the question and cite the specific source.
3. If the context does not contain information relevant to the question, clearly state: "I don't have information about that in my NASA mission documents."
4. NEVER fabricate or invent information, sources, or citations.
5. Maintain awareness of the conversation history to provide coherent follow-up responses.
6. When referencing previous answers, ensure consistency with cited sources."""


def generate_response(openai_key: str, user_message: str, context: str,
                      conversation_history: List[ResponseInputItemParam],
                      model: str = "gpt-3.5-turbo",
                      temperature: float = 0.1,
                      max_tokens: int = 200,
                      history_limit: int = 10) -> str:
    """Generate response using OpenAI with context and conversation history.

    Args:
        openai_key: OpenAI API key
        user_message: The user's current question
        context: Retrieved document context (XML formatted)
        conversation_history: List of previous conversation turns (role + content)
        model: OpenAI model to use
        temperature: Response creativity (0.0-1.0)
        max_tokens: Maximum response length
        history_limit: Number of recent messages to include (0 = all)

    Returns:
        Generated response text
    """

    user_prompt = f"""Based on the following retrieved NASA mission documents, answer the question.

Retrieved Context:
{context}

User Question: {user_message}

Instructions:
- Cite the specific mission and source from the document metadata in your answer
- Format citations as: (Source: [Mission Name] - [Document Type])
- If the context doesn't contain relevant information, say so clearly"""

    # Create a copy of conversation history to avoid mutation
    # Apply history limit if specified
    if history_limit > 0:
        messages = list(conversation_history[-history_limit:])
    else:
        messages = list(conversation_history)

    # Append current user message with context
    messages.append({"role": "user", "content": user_prompt})

    client = OpenAI(api_key=openai_key)

    response = client.responses.create(
        model=model,
        instructions=NASA_EXPERT_SYSTEM_PROMPT,
        input=messages,
        max_output_tokens=max_tokens,
        temperature=temperature
    )

    return response.output_text
