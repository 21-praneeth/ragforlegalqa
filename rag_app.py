from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import faiss
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# --- Environment Variables ---
os.environ["HF_TOKEN"] = "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"

# --- Load and preprocess the CSV ---
df = pd.read_csv("constitution.csv")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=128,
)

chunked_data = []

for _, row in df.iterrows():
    article_id = row["article_id"]
    article_text = str(row["article_desc"])
    full_text = f"{article_id}: {article_text}"
    chunks = text_splitter.split_text(full_text)
    for i, chunk in enumerate(chunks):
        chunked_data.append({
            "article_id": article_id,
            "chunk_id": i + 1,
            "chunk_text": chunk
        })

chunked_df = pd.DataFrame(chunked_data)

# --- Preprocessing ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

documents = [
    Document(
        page_content=preprocess_text(row["chunk_text"]),
        metadata={"article_id": row["article_id"], "chunk_id": row["chunk_id"]}
    )
    for _, row in chunked_df.iterrows()
]

# --- Embeddings ---
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    use_auth_token=False
)
texts = [doc.page_content for doc in documents]
embeddings = embedding_model.embed_documents(texts)

# --- FAISS index ---
embeddings_np = np.array(embeddings).astype("float32")
faiss.normalize_L2(embeddings_np)
index = faiss.IndexFlatIP(embeddings_np.shape[1])
index.add(embeddings_np)

# --- Load the Causal Language Model ---
model_name = "distilbert/distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- RAG Query Handler ---
def process_rag_query(user_query):
    cleaned_query = preprocess_text(user_query)
    query_embedding = embedding_model.embed_query(cleaned_query)
    query_embedding_np = np.array([query_embedding]).astype("float32")
    faiss.normalize_L2(query_embedding_np)

    D, I = index.search(query_embedding_np, k=5)

    relevant_chunks = []
    for score, idx in zip(D[0], I[0]):
        if score >= 0.4:
            relevant_chunks.append(documents[idx].page_content)

    if not relevant_chunks:
        return {"answer": "No, this does not come under legal queries."}

    context = "\n".join(relevant_chunks)
    prompt_text = (
        "Answer the following legal question based on the given context from the Constitution of India. "
        "Be concise and formal.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {user_query}\nAnswer:"
    )

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id
    )

    answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return {"answer": answer.strip()}

# --- Flask Setup ---
app = Flask(__name__)
CORS(app)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    query = data.get("query", "")
    result = process_rag_query(query)
    return jsonify(result)

if __name__ == "__main__":
    app.run(port=8000, host="0.0.0.0", debug=True)
