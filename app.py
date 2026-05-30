from flask import Flask, jsonify, render_template, request
from qa import answer_query_rag

app = Flask(__name__, template_folder="templates", static_folder="static")

EMBEDDING_BACKENDS = {
    "HuggingFace (local)": "huggingface",
    "Ollama (local)": "ollama",
    "OpenAI API": "openai"
}

LLM_BACKENDS = {
    "Mock (High-Fidelity)": "mock",
    "Ollama (tinyllama:chat)": "ollama",
    "OpenAI API": "openai"
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def query():
    payload = request.get_json(force=True)
    question = payload.get("question", "").strip()
    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    embedding_backend = EMBEDDING_BACKENDS.get(payload.get("embedding_backend", "HuggingFace (local)"), "huggingface")
    llm_backend = LLM_BACKENDS.get(payload.get("llm_backend", "Mock (High-Fidelity)"), "mock")
    k = int(payload.get("k", 3))
    hallucination_control = bool(payload.get("hallucination_control", True))
    confidence_threshold = float(payload.get("confidence_threshold", 0.75))
    openai_api_key = payload.get("openai_api_key") or None

    try:
        result = answer_query_rag(
            query=question,
            backend_embeddings=embedding_backend,
            backend_llm=llm_backend,
            k=k,
            hallucination_control=hallucination_control,
            confidence_threshold=confidence_threshold,
            openai_api_key=openai_api_key
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(result)

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", "8503"))
    app.run(host="0.0.0.0", port=port, debug=True)
