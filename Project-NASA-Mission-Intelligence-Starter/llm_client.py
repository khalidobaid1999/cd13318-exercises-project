import os

from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a NASA space mission expert assistant. You help users \
explore NASA missions using a curated library of mission documents.

Guidelines:
1. Ground every factual claim in the retrieved context provided below. The context \
is a list of numbered sources, each labelled with its mission, source file, and category.
2. When you use a source, cite it inline by its number, e.g. "(Source 2)".
3. If the context does not contain the answer, say so plainly and answer from general \
NASA knowledge only if you are confident, clearly marking it as background knowledge \
rather than something taken from the documents.
4. Never invent mission names, dates, crew members, or numbers. An honest "the documents \
do not cover that" is better than a plausible guess.
5. Use the conversation history to resolve follow-up questions and pronouns, but always \
re-check the retrieved context before answering.
6. Answer in clear prose, concise by default, expanding when the question calls for it. \
Use short lists for parallel items such as mission phases, crew, or instruments.
7. Keep a factual, engaging tone suited to someone curious about spaceflight."""

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-4o-mini") -> str:
    """Generate response using OpenAI with context"""

    system_prompt = SYSTEM_PROMPT
    messages = [{"role": "system", "content": system_prompt}]

    recent_history = conversation_history[-6:]
    if recent_history and recent_history[0].get("role") == "assistant":
        recent_history = recent_history[1:]
    messages.extend(recent_history)

    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": user_message})

    client = OpenAI(
        api_key=openai_key or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    response = client.chat.completions.create(messages=messages, 
                                              model=model, 
                                              temperature=0.7,
                                              max_tokens=800)

    answer = response.choices[0].message.content
    return answer.strip() if answer else ""
