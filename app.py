import os
from datetime import datetime, timezone

from flask import Flask, jsonify, request
from flask_cors import CORS
from openai import OpenAI
from langchain_rag import RAGAgent
import keycredentials
import get_response_rag
app = Flask(__name__)

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [
                "https://danish-ansari.com",
                "https://www.danish-ansari.com",
                "http://localhost:5500",
                "http://127.0.0.1:5500",
                "http://localhost:8000",
                "http://127.0.0.1:8000",
            ]
        }
    },
)

HF_TOKEN = keycredentials.hf_token
HF_MODEL = keycredentials.hf_model_llm
HF_MODEL_EMBEDDING = keycredentials.hf_model_embedding
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not set in the environment")

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)
from embedding import EmbeddingBuilder
from langchain_community.vectorstores import Chroma
embeddings = EmbeddingBuilder.build_embeddings(HF_MODEL_EMBEDDING, HF_TOKEN)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings, collection_name="internal_docs")
RAG_DOCUMENTS = [
    {
        "name": "generative_ai_sample_for_rag.pdf",
        "type": "PDF",
        "summary": "Compact Generative AI and RAG primer with headings, glossary terms, FAQ-style answers, and evaluation-oriented content.",
        "path": "./docs/generative_ai_sample_for_rag.pdf",
    },
    {
        "name": "sample.txt",
        "type": "TXT",
        "summary": "Sample operational workflow covering global inventory synchronization, extraction, security, loading, and troubleshooting steps.",
        "path": "./docs/sample.txt",
    },
    {
        "name": "LLM Powered Autonomous Agents - Html Page",
        "type": "WEB",
        "summary": "Overview of LLM-powered autonomous agents and their applications.",
        "path": "https://lilianweng.github.io/posts/2023-06-23-agent/"
    }
]

@app.get("/health")
def health():
    return jsonify(status="healthy"), 200

@app.get("/api/test")
def test_api():
    return jsonify(
        status="ok",
        source="lightsail-flask-api",
        message="Hello from Lightsail",
        hf_model=HF_MODEL,
        time_utc=datetime.now(timezone.utc).isoformat(),
    ), 200

@app.get("/api/rag-documents")
def rag_documents():
    return jsonify(
        status="ok",
        count=len(RAG_DOCUMENTS),
        documents=[
            {key: value for key, value in doc.items() if key != "path"}
            for doc in RAG_DOCUMENTS
        ],
        time_utc=datetime.now(timezone.utc).isoformat(),
    ), 200

@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify(
            status="error",
            error="message is required"
        ), 400

    try:
        completion = client.chat.completions.create(
            model=HF_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant for danish-ansari.com. "
                        "Reply in clean, professional, natural English. "
                        "Do not give multiple options unless asked. "
                        "Do not use markdown unless asked. "
                        "Keep responses concise and website-friendly."
                    ),
                },
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
            max_tokens=220,
            temperature=0.6,
        )

        reply = completion.choices[0].message.content or ""

        return jsonify(
            status="ok",
            source="hugging-face",
            model=HF_MODEL,
            reply=reply.strip(),
            time_utc=datetime.now(timezone.utc).isoformat(),
        ), 200

    except Exception as exc:
        return jsonify(
            status="error",
            source="hugging-face",
            error="HF call failed",
            details=str(exc),
            time_utc=datetime.now(timezone.utc).isoformat(),
        ), 502

@app.route('/api/ragqa', methods=['POST'])
def rag_qa():
    data = request.get_json(silent=True) or {}
    question = data.get('question', '').strip()

    if not question:
        return jsonify({'error': 'Question is required'}), 400

    try:
        #answer = rag_agent.rag_qa(question)
        answer = get_response_rag.rag_qa(question, vectorstore)
        return jsonify({
            'status': 'ok',
            'answer': answer,
            'documents_used': [doc['name'] for doc in RAG_DOCUMENTS],
            'time_utc': datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        app.logger.exception("Error while processing /api/ragqa")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5500, debug=False)
