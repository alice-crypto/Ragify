from flask import Blueprint, request, jsonify, render_template
from langchain.chains.question_answering import load_qa_chain
from app.models import Document
from app.db import db
from langchain.chains import RetrievalQA
import faiss
import numpy as np
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

import os

bp = Blueprint('main', __name__)

# Configure le modèle de langage (remplace par le modèle approprié)
# llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialisation de FAISS (remplace la dimension avec celle de ton modèle)
dim = 768
index = faiss.IndexFlatL2(dim)
docstore = {}
index_to_docstore_id = {}
embedding_model = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))
vector_store = FAISS(embedding_model.embed_query, index, docstore, index_to_docstore_id)


# Fonction pour ajouter un document à FAISS
def add_document_to_faiss(doc_id, content):
    embeddings = embedding_model.embed_query(content)  # Utiliser les embeddings OpenAI
    vector_store.add_texts([content], [doc_id])


# Fonction pour récupérer les documents
def get_relevant_docs(query):
    relevant_docs = Document.query.filter(Document.content.ilike(f'%{query}%')).all()
    return relevant_docs


# Fonction pour générer une réponse à partir des documents récupérés
def ask_question(query):
    relevant_docs = get_relevant_docs(query)

    # If no relevant documents are found, return a default response
    if not relevant_docs:
        return "No relevant documents found."

    # Create embeddings for the relevant documents
    embeddings = np.array([embedding_model.embed_query(doc.content) for doc in relevant_docs])

    # Add embeddings to FAISS index
    index.add(embeddings)  # This adds the vectors to the FAISS index

    # Load the QA chain
    qa_chain = load_qa_chain(llm=llm, chain_type="stuff")  # Change the chain type as needed

    # Use RetrievalQA with the retriever defined by the FAISS store
    qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=vector_store.as_retriever())

    # Execute the QA chain
    response = qa.run(query)
    return response


@bp.route('/')
def home():
    return render_template('index.html')


@bp.route('/add_phrase', methods=['POST'])
def add_phrase():
    content = request.form.get('content')
    if content:
        new_doc = Document(content=content)
        db.session.add(new_doc)
        db.session.commit()

        # Ajouter le document à FAISS
        add_document_to_faiss(new_doc.id, content)

        return jsonify({"message": "Phrase added"}), 201
    return jsonify({"error": "No content provided"}), 400


@bp.route('/list_phrases', methods=['GET'])
def list_phrases():
    docs = Document.query.all()
    return jsonify([doc.content for doc in docs]), 200


@bp.route('/ask', methods=['POST'])
def ask():
    question = request.json.get('question')
    if question:
        answer = ask_question(question)
        return jsonify({"answer": answer}), 200
    return jsonify({"error": "No question provided"}), 400
