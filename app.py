from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException
from qa import answer_query_rag
from settings import IS_RENDER

app = Flask(__name__, template_folder="templates", static_folder="static")

EMBEDDING_BACKENDS = {
    "HuggingFace (local)": "lexical",
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
    try:
        payload = request.get_json(force=True)
        question = payload.get("question", "").strip()
        if not question:
            return jsonify({"error": "Question cannot be empty."}), 400

        embedding_backend = EMBEDDING_BACKENDS.get(payload.get("embedding_backend", "HuggingFace (local)"), "lexical")
        llm_backend = LLM_BACKENDS.get(payload.get("llm_backend", "Mock (High-Fidelity)"), "mock")
        k = int(payload.get("k", 3))
        hallucination_control = bool(payload.get("hallucination_control", True))
        confidence_threshold = float(payload.get("confidence_threshold", 0.75))
        openai_api_key = payload.get("openai_api_key") or None
    except Exception as exc:
        return jsonify({"error": f"Invalid request payload: {exc}"}), 400

    fallback_notes = []
    if IS_RENDER and embedding_backend == "ollama":
        embedding_backend = "lexical"
        fallback_notes.append("Ollama embeddings are local-only on Render, so hosted lexical search was used.")
    if IS_RENDER and llm_backend == "ollama":
        llm_backend = "mock"
        fallback_notes.append("Ollama LLM is local-only on Render, so Mock answers were used.")

    if (embedding_backend == "openai" or llm_backend == "openai") and not openai_api_key:
        import os
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            if embedding_backend == "openai":
                embedding_backend = "lexical"
                fallback_notes.append("OpenAI embeddings need an API key, so hosted lexical search was used.")
            if llm_backend == "openai":
                llm_backend = "mock"
                fallback_notes.append("OpenAI LLM needs an API key, so Mock answers were used.")

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
        if fallback_notes:
            result["fallback_notes"] = fallback_notes
        result["active_backends"] = {
            "embeddings": embedding_backend,
            "llm": llm_backend
        }
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(result)

@app.errorhandler(HTTPException)
def handle_http_exception(exc):
    if request.path.startswith("/query") or request.path.startswith("/health"):
        return jsonify({"error": exc.description}), exc.code
    return exc

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", "8503"))
    app.run(host="0.0.0.0", port=port, debug=True)
