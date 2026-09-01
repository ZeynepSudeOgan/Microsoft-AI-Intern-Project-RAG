from flask import Flask, render_template, request, jsonify

from foundry_local_sdk import FoundryLocalManager
from retrieve import get_top_chunks

# retrieve.py already initializes the FoundryLocalManager singleton and
# loads the embedding model on import — reuse that instance here.
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

RELEVANCE_THRESHOLD = 0.4
FALLBACK_ANSWER = "I don't have that information in my documents."

app = Flask(__name__)


def answer_query(user_question):
    """Retrieve relevant context and generate an answer using the local LLM.

    Returns a dict with the answer text and the source chunks used,
    so the frontend can display retrieval transparency (like footnotes).
    """
    top_chunks = get_top_chunks(user_question, k=3)

    if not top_chunks or top_chunks[0][0] < RELEVANCE_THRESHOLD:
        return {"answer": FALLBACK_ANSWER, "sources": []}

    context = "\n".join(f"- {content}" for score, content in top_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {user_question}"

    response = client.complete_chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ])

    answer = response.choices[0].message.content
    sources = [
        {"content": content, "score": round(score, 3)}
        for score, content in top_chunks
    ]
    return {"answer": answer, "sources": sources}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    result = answer_query(question)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)