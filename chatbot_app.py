import re
from typing import List, Optional

import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document


def inject_chat_styles() -> None:
    """
    Inject custom CSS to give a more polished, ChatGPT‑like look.
    """
    st.markdown(
        """
        <style>
        /* App background */
        .stApp {
            background: radial-gradient(circle at top, #202b3a 0, #050816 55%, #000000 100%) !important;
            color: #e5e7eb !important;
        }

        /* Center main content */
        .main-container {
            max-width: 900px;
            margin: 0 auto;
            padding: 1.5rem 0 4rem 0;
        }

        /* Header title */
        .app-title {
            font-size: 2.1rem;
            font-weight: 700;
            background: linear-gradient(90deg, #22c55e, #38bdf8);
            -webkit-background-clip: text;
            color: transparent;
            text-align: center;
            margin-bottom: 0.15rem;
        }

        .app-subtitle {
            text-align: center;
            font-size: 0.95rem;
            color: #9ca3af;
            margin-bottom: 1.5rem;
        }

        /* Chat input bar */
        .stChatInput {
            border-top: 1px solid rgba(148, 163, 184, 0.35);
            background: linear-gradient(to top, rgba(15, 23, 42, 0.98), rgba(15, 23, 42, 0.92));
            backdrop-filter: blur(12px);
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: radial-gradient(circle at top left, #020617 0, #020617 45%, #000000 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.35);
        }

        /* Slight rounding for chat messages */
        [data-testid="stChatMessage"] {
            border-radius: 12px;
            padding: 0.35rem 0.35rem;
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            font-size: 0.82rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def extract_video_id(url_or_id: str) -> Optional[str]:
    """
    Accept either a raw YouTube video ID or a full URL and return the video ID.
    """
    url_or_id = url_or_id.strip()

    # If it already looks like an ID (11 chars, letters/numbers/_/-), just return it
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", url_or_id):
        return url_or_id

    # Try to extract from common URL formats (watch, youtu.be, embed, shorts, etc.)
    patterns = [
        r"[?&]v=([0-9A-Za-z_-]{11})",          # https://www.youtube.com/watch?v=VIDEO_ID&...
        r"(?:youtu\.be/)([0-9A-Za-z_-]{11})",  # https://youtu.be/VIDEO_ID
        r"(?:embed/)([0-9A-Za-z_-]{11})",      # https://www.youtube.com/embed/VIDEO_ID
        r"(?:shorts/)([0-9A-Za-z_-]{11})",     # https://www.youtube.com/shorts/VIDEO_ID
    ]

    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)

    # Fallback: grab the first 11-char token that looks like a video id
    fallback = re.search(r"([0-9A-Za-z_-]{11})", url_or_id)
    if fallback:
        return fallback.group(1)

    return None


def fetch_transcript_text(video_id: str, languages: Optional[List[str]] = None) -> str:
    """
    Fetch the YouTube transcript and return it as a single text string.
    Handles both classmethod and instance-method styles, depending on library version.
    """
    if languages is None:
        languages = ["en"]

    try:
        # Some versions expose a classmethod `get_transcript`, others use an instance with `.fetch`
        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            transcript_chunks = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        else:
            api = YouTubeTranscriptApi()
            # Older style used in your notebook: ytt.fetch(video_id, languages=['en'])
            if hasattr(api, "fetch"):
                transcript_chunks = api.fetch(video_id, languages=languages)
            else:
                raise AttributeError(
                    "youtube-transcript-api does not provide get_transcript or fetch methods."
                )
    except Exception as e:
        # Let caller show a clear error message
        raise RuntimeError(f"Could not fetch transcript for video '{video_id}': {e}") from e

    # transcript_chunks can be:
    # - list of dicts: {"text": "...", "start": ..., "duration": ...}
    # - list of objects with a `.text` attribute (FetchedTranscriptSnippet in your notebook)
    parts: List[str] = []
    for chunk in transcript_chunks:
        if hasattr(chunk, "text"):
            parts.append(getattr(chunk, "text", ""))
        elif isinstance(chunk, dict):
            parts.append(chunk.get("text", ""))
    full_text = " ".join(parts)
    return full_text


def build_vector_store(text: str):
    """
    Build a FAISS vector store over the given text using Ollama embeddings.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    documents: List[Document] = splitter.create_documents([text])

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store


def get_retriever_from_video(video_url_or_id: str):
    video_id = extract_video_id(video_url_or_id)
    if not video_id:
        raise ValueError("Could not extract a valid YouTube video ID from the input.")

    transcript_text = fetch_transcript_text(video_id)
    if not transcript_text.strip():
        raise ValueError("No transcript available for this video (may be disabled or missing).")

    vector_store = build_vector_store(transcript_text)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return retriever, transcript_text


def build_llm():
    return ChatOllama(model="tinyllama:chat")


PROMPT_TEMPLATE = PromptTemplate(
    template=(
        "You are a helpful assistant.\n"
        "Use the chat history and the provided context from the YouTube video.\n"
        "Answer the user's question **only** using that context.\n"
        "If the answer is not in the context, say \"I don't know based on this video\".\n\n"
        "Chat history:\n"
        "{history}\n\n"
        "Context:\n"
        "{context}\n\n"
        "Question:\n"
        "{question}\n\n"
        "Answer:"
    ),
    input_variables=["context", "question", "history"],
)


def answer_question(retriever, llm, question: str, history_text: str) -> str:
    docs: List[Document] = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt_value = PROMPT_TEMPLATE.invoke(
        {"context": context, "question": question, "history": history_text}
    )
    response = llm.invoke(prompt_value)
    return response.content


def main():
    st.set_page_config(page_title="YouTube RAG Chatbot", page_icon="🤖", layout="wide")
    inject_chat_styles()

    # Top-level header area
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="app-title">YouTube RAG Chatbot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Paste a YouTube link, then chat with an AI grounded in that video\'s transcript.</div>',
        unsafe_allow_html=True,
    )

    if "retriever" not in st.session_state:
        st.session_state.retriever = None
        st.session_state.video_text = ""
    if "messages" not in st.session_state:
        # chat history: list of {"role": "user"|"assistant", "content": str}
        st.session_state.messages = []

    with st.sidebar:
        st.header("Video settings")
        video_input = st.text_input(
            "YouTube URL or Video ID",
            placeholder="https://www.youtube.com/watch?v=ukzFI9rgwfU",
        )
        load_video = st.button("Load video")

        if load_video:
            if not video_input.strip():
                st.error("Please paste a YouTube URL or video ID.")
            else:
                with st.spinner("Fetching transcript and building knowledge base..."):
                    try:
                        retriever, transcript_text = get_retriever_from_video(video_input)
                    except Exception as e:
                        st.session_state.retriever = None
                        st.session_state.video_text = ""
                        st.error(f"Error loading video: {e}")
                    else:
                        st.session_state.retriever = retriever
                        st.session_state.video_text = transcript_text
                        st.session_state.messages = []  # reset chat for new video
                        st.success("Video loaded successfully! You can now ask questions.")

    if st.session_state.retriever is None:
        st.info("Load a YouTube video from the sidebar to start chatting.")
        return

    llm = build_llm()

    st.subheader("Chat about the video")

    # show existing chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # chat input
    user_question = st.chat_input("Ask something about this video")

    if user_question:
        # add user message to history and display
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # build short text history for the LLM
        history_lines: List[str] = []
        for msg in st.session_state.messages[:-1]:  # exclude current question
            prefix = "User" if msg["role"] == "user" else "Assistant"
            history_lines.append(f"{prefix}: {msg['content']}")
        history_text = "\n".join(history_lines[-8:])  # last few turns

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = answer_question(
                        st.session_state.retriever, llm, user_question, history_text
                    )
                except Exception as e:
                    st.error(f"Error while generating answer: {e}")
                    return
                st.markdown(answer)

        # save assistant reply
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # optional: show retrieved snippets for the last question
        with st.expander("Show transcript snippets used for the last answer"):
            docs: List[Document] = st.session_state.retriever.invoke(user_question)
            for i, doc in enumerate(docs, start=1):
                st.markdown(f"**Chunk {i}:**")
                st.write(doc.page_content)
                st.markdown("---")

    # close main-container div
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

