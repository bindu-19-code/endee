import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity
import os

api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=api_key)

embed_model = SentenceTransformer('paraphrase-MiniLM-L3-v2', device='cpu')
def load_documents(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        docs = f.readlines()
    docs = [doc.strip() for doc in docs if doc.strip() != ""]
    return docs

base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, "dataset.txt")

documents = load_documents(file_path)
doc_embeddings = embed_model.encode(documents)

def retrieve(query, top_k=3):
    query_embedding = embed_model.encode([query])
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [documents[i] for i in top_indices]

def generate_answer(query, contexts):
    context_text = "\n".join(contexts)

    prompt = f"""
Answer ONLY using the context below.

Context:
{context_text}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

st.title("🤖 AI Study Assistant")

query = st.text_input("Ask your question:")

if st.button("Get Answer"):
    with st.spinner("Thinking..."):
        contexts = retrieve(query)
        answer = generate_answer(query, contexts)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Contexts")
    for c in contexts:
        st.write("- " + c)