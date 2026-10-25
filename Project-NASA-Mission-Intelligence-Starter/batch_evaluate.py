import argparse
import json
from typing import Any, Dict, List

import rag_client
import llm_client
import ragas_evaluator

def evaluate_question(collection, item: Dict[str, Any], n_results: int, model: str) -> Dict[str, Any]:
    """Retrieve, answer, and score a single test question"""
    question = item["question"]
    docs = rag_client.retrieve_documents(collection, question, n_results)
    contexts = docs["documents"][0]
    context = rag_client.format_context(contexts, docs["metadatas"][0])

    answer = llm_client.generate_response(None, question, context, [], model)
    scores = ragas_evaluator.evaluate_response_quality(question, answer, contexts, reference=item.get("reference"))
    return {**item, "answer": answer, "scores": scores}

def summarize(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Aggregate each numeric metric across all questions"""
    values: Dict[str, List[float]] = {}
    for result in results:
        for name, score in result["scores"].items():
            if isinstance(score, (int, float)):
                values.setdefault(name, []).append(score)

    return {
        name: {"mean": sum(v) / len(v), "min": min(v), "max": max(v), "count": len(v)}
        for name, v in values.items()
    }

def main():
    parser = argparse.ArgumentParser(description='Batch RAGAS evaluation of the NASA RAG system')
    parser.add_argument('--questions', default='test_questions.json', help='Path to the test question set')
    parser.add_argument('--chroma-dir', default='./chroma_db_openai', help='ChromaDB persist directory')
    parser.add_argument('--collection-name', default='nasa_space_missions_text', help='Collection name')
    parser.add_argument('--n-results', type=int, default=3, help='Documents to retrieve per question')
    parser.add_argument('--model', default='gpt-4o-mini', help='OpenAI model for answers')
    parser.add_argument('--output', help='Optional path to save per-question results and the summary as JSON')
    args = parser.parse_args()

    with open(args.questions, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    collection = rag_client.initialize_rag_system(args.chroma_dir, args.collection_name)
    print(f"Evaluating {len(questions)} questions against {args.chroma_dir}/{args.collection_name} "
          f"({collection.count()} documents)\n")

    results = []
    for i, item in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] ({item.get('category', '-')}, {item.get('mission', '-')}) {item['question']}")
        try:
            result = evaluate_question(collection, item, args.n_results, args.model)
        except Exception as e:
            result = {**item, "answer": "", "scores": {"error": f"Pipeline failed: {e}"}}
        results.append(result)

        scores = result["scores"]
        if "error" in scores:
            print(f"    ERROR: {scores['error']}")
        else:
            print("    " + " | ".join(f"{k} {v:.3f}" for k, v in scores.items() if isinstance(v, float)))
            if "warnings" in scores:
                print(f"    warnings: {scores['warnings']}")
        print()

    summary = summarize(results)
    print("=" * 60)
    print("AGGREGATE SCORES")
    print("=" * 60)
    print(f"{'metric':<22}{'mean':>8}{'min':>8}{'max':>8}{'n':>5}")
    for name, s in summary.items():
        print(f"{name:<22}{s['mean']:>8.3f}{s['min']:>8.3f}{s['max']:>8.3f}{s['count']:>5}")
    failed = sum(1 for r in results if "error" in r["scores"])
    print(f"\nQuestions evaluated: {len(results) - failed}/{len(results)}")

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump({"results": results, "summary": summary}, f, indent=2)
        print(f"Saved results to {args.output}")

if __name__ == "__main__":
    main()
