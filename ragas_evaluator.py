"""
RAGAS Evaluator for RAG Response Quality
Evaluates responses using Faithfulness, ResponseRelevancy, and ContextPrecision metrics.
"""

import os
import asyncio
from typing import Dict, List

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithoutReference
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """
    Evaluate response quality using RAGAS metrics.

    Args:
        question: The user's question
        answer: The RAG system's answer
        contexts: List of retrieved context strings

    Returns:
        Dictionary with metric scores (0-1) or error message
    """
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    if not contexts:
        return {"error": f"No contexts provided for evaluation"}

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        # Try alternative environment variable used by chat.py
        openai_api_key = os.environ.get("CHROMA_OPENAI_API_KEY")

    if not openai_api_key:
        return {"error": "OpenAI API key not found"}

    try:
        # Create evaluator LLM with gpt-3.5-turbo
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=openai_api_key
        ))

        # Create evaluator embeddings with text-embedding-3-small
        langchain_openai_embeddings = OpenAIEmbeddings(
            api_key=openai_api_key,
            model="text-embedding-3-small"
        )
        evaluator_embeddings = LangchainEmbeddingsWrapper(langchain_openai_embeddings)

        # Define metric instances
        faithfulness_metric = Faithfulness(llm=evaluator_llm)
        relevancy_metric = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
        precision_metric = LLMContextPrecisionWithoutReference(llm=evaluator_llm)

        # Create evaluation sample
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts
        )

        # Run async evaluation
        async def run_evaluation():
            faith_score = await faithfulness_metric.single_turn_ascore(sample)
            relevancy_score = await relevancy_metric.single_turn_ascore(sample)
            precision_score = await precision_metric.single_turn_ascore(sample)
            return faith_score, relevancy_score, precision_score

        scores = asyncio.run(run_evaluation())

        # Return evaluation results
        return {
            "faithfulness": round(scores[0], 4) if scores[0] is not None else 0.0,
            "response_relevancy": round(scores[1], 4) if scores[1] is not None else 0.0,
            "context_precision": round(scores[2], 4) if scores[2] is not None else 0.0
        }

    except Exception as e:
        return {"error": f"Evaluation failed: {str(e)}"}
