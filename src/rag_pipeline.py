import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate


# 1. Project configuration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_ROOT / "storage" / "chroma_db"

load_dotenv(PROJECT_ROOT / ".env")

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN not found"
    )


# 2. Initialize Hugging Face embedding client

embedding_client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN,
)


# 3. Load existing ChromaDB collection

if not CHROMA_DIR.exists():
    raise FileNotFoundError(
        f"ChromaDB directory not found: {CHROMA_DIR}"
    )

vector_store = Chroma(
    collection_name="maintenance_manuals",
    embedding_function=None,
    persist_directory=str(CHROMA_DIR),
)

document_count = vector_store._collection.count()

if document_count == 0:
    raise ValueError(
        "The maintenance_manuals collection is empty."
    )


# 4. Initialize LLM

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    provider="featherless-ai",
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=512,
    temperature=0.2,
)

model = ChatHuggingFace(llm=llm)


# 5. Define the RAG prompt

rag_prompt = ChatPromptTemplate.from_template(
    """
You are an Industrial Maintenance Copilot.

Answer the user's question using only the maintenance
manual excerpts provided below.

Rules:
- Use the excerpts as your factual source.
- Do not invent technical specifications or procedures.
- If the excerpts do not contain enough information,
  clearly say that the information is not available
  in the provided excerpts.
- Give a clear, practical answer.
- Cite relevant source filenames and page numbers.
- Do not claim a recommendation comes from a manual
  unless the excerpts support it.

Maintenance manual excerpts:
{context}

User question:
{question}

Answer:
"""
)


# 6. Retrieve relevant chunks from ChromaDB

def retrieve_context(
    question: str,
    k: int = 4,
) -> list[dict]:
    """Retrieve relevant chunks."""

    query_embedding = embedding_client.feature_extraction(
        question,
        model="BAAI/bge-small-en-v1.5",
    )

    results = vector_store._collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    retrieved_chunks = []

    if not results["documents"] or not results["documents"][0]:
        return retrieved_chunks

    for i, text in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]

        retrieved_chunks.append(
            {
                "text": text,
                "source": metadata["source"],
                "page": metadata["page"],
                "chunk_index": metadata["chunk_index"],
                "distance": results["distances"][0][i],
            }
        )

    return retrieved_chunks


# 7. Generate an answer

def answer_question(
    query: str,
    top_k: int = 3,
) -> dict:
    """Retrieve context and generate a answer."""

    results = retrieve_context(query, k=top_k)

    if not results:
        return {
            "question": query,
            "answer": (
                "I could not retrieve relevant excerpts from "
                "the maintenance manuals for this question."
            ),
            "sources": [],
        }

    context_parts = []

    for i, result in enumerate(results, start=1):
        context_parts.append(
            f"""[Source {i}]
Filename: {result['source']}
Page: {result['page']}
Content:
{result['text']}"""
        )

    context = "\n\n".join(context_parts)

    messages = rag_prompt.invoke(
        {
            "context": context,
            "question": query,
        }
    )

    response = model.invoke(messages)

    return {
        "question": query,
        "answer": response.content,
        "sources": results,
    }