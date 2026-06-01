import os
import sys
import logging

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException
from qa import answer_query_rag

app = Flask(__name__, template_folder="templates", static_folder="static")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)

EMBEDDING_BACKENDS = {
    "Hosted search": "lexical",
}

LLM_BACKENDS = {
    "Mock (High-Fidelity)": "mock",
    "OpenAI API": "openai",
    "Grok API": "grok"
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def query():
    try:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return error_response("Request body must be valid JSON.", 400)

        question = payload.get("question", "").strip()
        if not question:
            return error_response("Question cannot be empty.", 400)

        embedding_backend = EMBEDDING_BACKENDS.get(payload.get("embedding_backend", "Hosted search"), "lexical")
        llm_backend = LLM_BACKENDS.get(payload.get("llm_backend", os.getenv("DEFAULT_LLM_BACKEND", "Grok API")), "grok")
        k = max(1, min(int(payload.get("k", 3)), 8))
        hallucination_control = parse_bool(payload.get("hallucination_control", True))
        confidence_threshold = max(0.2, min(float(payload.get("confidence_threshold", 0.75)), 1.5))
        openai_api_key = payload.get("openai_api_key") or None
        xai_api_key = payload.get("xai_api_key") or None
    except Exception as exc:
        app.logger.exception("Invalid request payload")
        return error_response(f"Invalid request payload: {exc}", 400)

    fallback_notes = []
    if (embedding_backend == "openai" or llm_backend == "openai") and not openai_api_key:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            if embedding_backend == "openai":
                embedding_backend = "lexical"
                fallback_notes.append("OpenAI embeddings need an API key, so hosted lexical search was used.")
            if llm_backend == "openai":
                llm_backend = "mock"
                fallback_notes.append("OpenAI LLM needs an API key, so Mock answers were used.")
    if llm_backend == "grok" and not xai_api_key:
        xai_api_key = os.getenv("XAI_API_KEY")
        if not xai_api_key:
            llm_backend = "mock"
            fallback_notes.append("Grok answers need an xAI API key, so Mock answers were used.")

    try:
        app.logger.info(
            "Handling query with embeddings=%s llm=%s k=%s hallucination_control=%s",
            embedding_backend,
            llm_backend,
            k,
            hallucination_control,
        )
        result = answer_query_rag(
            query=question,
            backend_embeddings=embedding_backend,
            backend_llm=llm_backend,
            k=k,
            hallucination_control=hallucination_control,
            confidence_threshold=confidence_threshold,
            openai_api_key=openai_api_key,
            xai_api_key=xai_api_key
        )
        if fallback_notes:
            result["fallback_notes"] = fallback_notes
        result["active_backends"] = {
            "embeddings": embedding_backend,
            "llm": llm_backend
        }
    except Exception as exc:
        app.logger.exception("Query failed")
        return error_response("Query failed. Check Render logs for details.", 500, detail=str(exc))

    return jsonify(result)

def error_response(message: str, status_code: int, detail: str = None):
    payload = {"error": message, "status": status_code}
    if detail and os.getenv("SHOW_ERROR_DETAILS", "false").lower() == "true":
        payload["detail"] = detail
    return jsonify(payload), status_code

def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "0", "no", "off"}
    return bool(value)

@app.errorhandler(HTTPException)
def handle_http_exception(exc):
    if request.path.startswith("/query") or request.path.startswith("/health") or request.path.startswith("/api"):
        return error_response(exc.description, exc.code)
    return exc

@app.errorhandler(Exception)
def handle_unexpected_exception(exc):
    app.logger.exception("Unhandled server error")
    return error_response("Internal server error. Check Render logs for details.", 500, detail=str(exc))

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", "8503"))
    app.run(host="0.0.0.0", port=port, debug=True)
