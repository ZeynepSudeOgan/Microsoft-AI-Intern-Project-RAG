from foundry_local_sdk import FoundryLocalManager
from retrieve import get_top_chunks

# retrieve.py already calls FoundryLocalManager.initialize() on import,
# so here we just grab the existing singleton instance instead of
# initializing it again (that would raise FoundryLocalException).
manager = FoundryLocalManager.instance

chat_model = manager.catalog.get_model("phi-3.5-mini")

if not chat_model.is_cached:
    print("Downloading chat model...")
    chat_model.download(lambda p: print(f"\r%{p:.0f}", end="", flush=True))
    print()

chat_model.load()

client = chat_model.get_chat_client()

SYSTEM_PROMPT = (
    "You are a strict Q&A assistant. You must answer using ONLY the "
    "information in the provided context below. Never use outside "
    "knowledge, even if you know the answer. Never add facts, examples, "
    "or details that are not supported by the context.\n\n"
    "You may reason using the ideas and meaning of the context, not just "
    "its exact wording — for example, if the context describes a concept "
    "without using a specific term, you may still connect it to a "
    "question that uses that term, as long as the substance of the "
    "answer comes from the context.\n\n"
    "If the context genuinely does not contain information relevant to "
    "the question, your ENTIRE response must be exactly this sentence, "
    "with no explanation, reasoning, or preamble before or after it: "
    "\"I don't have that information in my documents.\"\n\n"
    "Keep answers concise and grounded only in the given context."
)

# If the best-matching chunk's similarity score is below this, the context
# is considered irrelevant and we skip the LLM call entirely — a safety net
# in case the model ignores the system prompt and hallucinates anyway.
RELEVANCE_THRESHOLD = 0.4
FALLBACK_ANSWER = "I don't have that information in my documents."


def answer_query(user_question):
    """Retrieve relevant context and generate an answer using the local LLM."""
    top_chunks = get_top_chunks(user_question, k=3)

    # Safety net: if even the best match is a weak similarity score,
    # don't bother asking the LLM — it's likely to hallucinate anyway.
    if not top_chunks or top_chunks[0][0] < RELEVANCE_THRESHOLD:
        return FALLBACK_ANSWER

    context = "\n".join(f"- {content}" for score, content in top_chunks)

    prompt = f"Context:\n{context}\n\nQuestion: {user_question}"

    response = client.complete_chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ])

    return response.choices[0].message.content


def main():
    print("Local RAG Assistant — type 'exit' to quit\n")
    while True:
        question = input("Soru: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        answer = answer_query(question)
        print(f"Cevap: {answer}\n")

    chat_model.unload()


if __name__ == "__main__":
    main()