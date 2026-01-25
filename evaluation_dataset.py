"""
Batch Evaluation Workflow for RAG Pipeline
Runs the full RAG pipeline (retrieve -> generate -> evaluate) on test questions.
"""

import json
import os
from typing import List, Dict, Any
from statistics import mean

from dotenv import load_dotenv

from rag_client import discover_chroma_backends, initialize_rag_system, retrieve_documents, format_context
from llm_client import generate_response
from ragas_evaluator import evaluate_response_quality

load_dotenv()


def load_test_questions(test_file: str = "test_questions.json") -> List[Dict[str, str]]:
    """Load test questions from JSON file."""
    with open(test_file, "r") as f:
        return json.load(f)


def run_batch_evaluation(test_file: str = "test_questions.json") -> List[Dict[str, Any]]:
    """
    Load questions, run full RAG pipeline, compute metrics for each.

    Args:
        test_file: Path to JSON file containing test questions

    Returns:
        List of result dictionaries with question, answer, contexts, and metrics
    """
    # Load test questions
    questions = load_test_questions(test_file)
    print(f"Loaded {len(questions)} test questions from {test_file}")

    # Initialize RAG system
    backends = discover_chroma_backends()
    if not backends:
        print("ERROR: No ChromaDB backends found!")
        return []

    # Use first available backend
    backend_key = list(backends.keys())[0]
    backend_info = backends[backend_key]
    print(f"Using backend: {backend_info['display_name']} ({backend_info['document_count']} documents)")

    collection, success, error = initialize_rag_system(
        backend_info["directory"],
        backend_info["collection_name"]
    )

    if not success or collection is None:
        print(f"ERROR: Failed to initialize RAG system: {error}")
        return []

    # Get OpenAI API key
    openai_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("CHROMA_OPENAI_API_KEY")
    if not openai_key:
        print("ERROR: OpenAI API key not found in environment")
        return []

    results = []

    # Process each question through the full pipeline
    for idx, item in enumerate(questions, start=1):
        question = item["question"]
        print(f"\nProcessing question {idx}/{len(questions)}:{question}")
        
        # Step 1: Retrieve documents
        query_result = retrieve_documents(collection, question, n_results=3)

        documents = query_result.get("documents", [[]])[0] if query_result else []
        metadatas = query_result.get("metadatas", [[]])[0] if query_result else []

        if not documents:
            print(f"  WARNING: No documents retrieved for question {idx}")
            results.append({
                "question": question,
                "answer": "No documents found",
                "contexts": [],
                "metrics": {"error": "No documents retrieved"}
            })
            continue

        # Step 2: Format context and generate response
        context = format_context(documents, metadatas)

        answer = generate_response(
            openai_key=openai_key,
            user_message=question,
            context=context,
            conversation_history=[],
            model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=200
        )

        # Step 3: Evaluate response quality
        metrics = evaluate_response_quality(
            question=question,
            answer=answer,
            contexts=documents
        )

        results.append({
            "question": question,
            "answer": answer,
            "contexts": documents,
            "metrics": metrics
        })

        # Print per-question result
        if "error" not in metrics:
            print(f"  Faithfulness={metrics['faithfulness']:.2f}, "
                  f"Relevancy={metrics['response_relevancy']:.2f}, "
                  f"Precision={metrics['context_precision']:.2f}")
        else:
            print(f"  Error: {metrics['error']}")

    return results


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print per-question results and aggregate statistics."""

    print("\n" + "=" * 80)
    print("BATCH EVALUATION RESULTS")
    print("=" * 80)

    # Per-question results
    for idx, result in enumerate(results, start=1):
        print(f"\nQuestion {idx}: {result['question']}")

        print(f"Answer: { result['answer']}")

        metrics = result['metrics']
        if "error" in metrics:
            print(f"Metrics: Error - {metrics['error']}")
        else:
            print(f"Metrics: Faithfulness={metrics['faithfulness']:.2f}, "
                  f"Relevancy={metrics['response_relevancy']:.2f}, "
                  f"Precision={metrics['context_precision']:.2f}")

    # Aggregate statistics
    print("\n" + "=" * 80)
    print("AGGREGATE SUMMARY")
    print("=" * 80)

    # Filter out results with errors
    valid_results = [r for r in results if "error" not in r["metrics"]]

    if not valid_results:
        print("No valid results to aggregate.")
        return

    print(f"Total Questions: {len(results)}")
    print(f"Successfully Evaluated: {len(valid_results)}")

    # Calculate aggregate stats for each metric
    metric_names = ["faithfulness", "response_relevancy", "context_precision"]
    display_names = ["Faithfulness", "Response Relevancy", "Context Precision"]

    for metric_key, display_name in zip(metric_names, display_names):
        values = [r["metrics"][metric_key] for r in valid_results]
        avg = mean(values)
        min_val = min(values)
        max_val = max(values)
        print(f"{display_name:20s} Mean={avg:.2f}, Min={min_val:.2f}, Max={max_val:.2f}")

    print("=" * 80)


def main():
    """Main entry point for batch evaluation."""
    print("=" * 80)
    print("RAG BATCH EVALUATION WORKFLOW")
    print("=" * 80)

    results = run_batch_evaluation("test_questions.json")

    if results:
        print_summary(results)
    else:
        print("\nNo results to display.")


if __name__ == "__main__":
    main()
