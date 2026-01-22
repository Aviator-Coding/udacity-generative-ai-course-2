from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional
import os
# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics.collections import BleuScore, Faithfulness, RougeScore
    from ragas.metrics import NonLLMContextPrecisionWithReference, ResponseRelevancy,
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}
    openai_api_key = os.environ("OPENAI_API_KEY")
    # TODO: Create evaluator LLM with model gpt-3.5-turbo
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=openai_api_key,
            # base_url="https://openai.vocareum.com/v1"
        ))
    # TODO: Create evaluator_embeddings with model test-embedding-3-small
    langchain_openai_embeddings = OpenAIEmbeddings(
            api_key=openai_api_key,
            model="test-embedding-3-small"
            # openai_api_base="https://openai.vocareum.com/v1"
        )
    evaluator_embeddings = LangchainEmbeddingsWrapper(langchain_openai_embeddings)


    # TODO: Define an instance for each metric to evaluate
    # TODO: Evaluate the response using the metrics
    # TODO: Return the evaluation results

    pass
