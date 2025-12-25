from flask import Flask, render_template, request, jsonify
import os

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

app = Flask(__name__)

# -------------------------------
# 1. Load FAISS
# -------------------------------
def load_faiss():
    embedding = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"}
    )

    index_path = "research/faiss_index"
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}")

    db = FAISS.load_local(
        index_path,
        embedding,
        allow_dangerous_deserialization=True
    )
    return db

db = load_faiss()

# -------------------------------
# 2. Build RAG
# -------------------------------
retriever = db.as_retriever(search_kwargs={"k": 8})
llm = Ollama(model="mistral", temperature=0.3)

prompt = PromptTemplate(
    template="""
You are a medical assistant. Answer using ONLY the context below.
Provide a detailed and comprehensive explanation.
If the answer is not in the context, say "I don't know".

--- CONTEXT ---
{context}
----------------

Question: {question}

Answer:
""",
    input_variables=["context", "question"]
)

def rag_answer(question):
    docs = retriever.get_relevant_documents(question)
    context = "\n".join([d.page_content for d in docs])
    final_prompt = prompt.format(context=context, question=question)
    return llm.invoke(final_prompt)

# -------------------------------
# 3. Routes
# -------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": ""})

    answer = rag_answer(question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
