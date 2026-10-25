from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import BleuScore, NonLLMContextPrecisionWithReference, ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}
    
    base_url = os.getenv("OPENAI_BASE_URL")
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", base_url=base_url))
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small", base_url=base_url)
    )

    metrics = {
        "response_relevancy": ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        "faithfulness": Faithfulness(llm=evaluator_llm),
    }

    sample = SingleTurnSample(user_input=question, response=answer, retrieved_contexts=contexts)
    return {name: metric.single_turn_score(sample) for name, metric in metrics.items()}
