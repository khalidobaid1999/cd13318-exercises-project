from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional
import math
import os
from dotenv import load_dotenv

load_dotenv()

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import BleuScore, LLMContextPrecisionWithReference, ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

def validate_inputs(question, answer, contexts, reference=None) -> Optional[str]:
    """Return a clear error message for empty or malformed inputs, or None if they are valid"""
    if not isinstance(question, str) or not question.strip():
        return "Question must be a non-empty string"
    if not isinstance(answer, str) or not answer.strip():
        return "Answer must be a non-empty string"
    if not isinstance(contexts, list) or not contexts:
        return "No retrieved contexts to evaluate against (expected a non-empty list of strings)"
    if not all(isinstance(c, str) and c.strip() for c in contexts):
        return "Every retrieved context must be a non-empty string"
    if reference is not None and (not isinstance(reference, str) or not reference.strip()):
        return "Reference answer, when given, must be a non-empty string"
    return None

def evaluate_response_quality(question: str, answer: str, contexts: List[str],
                              reference: Optional[str] = None) -> Dict[str, float]:
    """
    Evaluate response quality using RAGAS metrics
    """
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    input_error = validate_inputs(question, answer, contexts, reference)
    if input_error:
        return {"error": input_error}

    base_url = os.getenv("OPENAI_BASE_URL")
    # gpt-3.5-turbo is DEPRECATED and returns 404, so I used gpt-4o-mini
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", base_url=base_url))
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small", base_url=base_url)
    )

    metrics = {
        "response_relevancy": ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        "faithfulness": Faithfulness(llm=evaluator_llm),
    }
    if reference:
        metrics.update({
            "context_precision": LLMContextPrecisionWithReference(llm=evaluator_llm),
            "bleu_score": BleuScore(),
            "rouge_score": RougeScore(),
        })

    sample = SingleTurnSample(user_input=question, response=answer,
                              retrieved_contexts=contexts, reference=reference)

    scores, warnings = {}, []
    for name, metric in metrics.items():
        try:
            score = float(metric.single_turn_score(sample))
        except Exception as e:
            warnings.append(f"{name} failed: {e}")
            continue
        if math.isnan(score):
            warnings.append(f"{name} could not be computed for this answer")
        else:
            scores[name] = score

    if not scores:
        return {"error": "; ".join(warnings)}
    if warnings:
        scores["warnings"] = "; ".join(warnings)
    return scores
