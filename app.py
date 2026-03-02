from flask import Flask, render_template, jsonify, request
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI  # You can replace this with a local model like Llama
import os

app = Flask(__name__)

# 1. Load the Memory (Vectorstore)
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
db = Chroma(persist_directory="./vectorstore", embedding_function=embeddings)

# 2. Setup the Retriever
retriever = db.as_retriever(search_kwargs={"k": 2})

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    # This searches the Gale Encyclopedia for the answer
    result = db.similarity_search(input)
    return str(result[0].page_content)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)