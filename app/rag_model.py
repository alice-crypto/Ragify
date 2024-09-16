from langchain import OpenAI
from langchain.chains import RetrievalQA
from app.db import get_relevant_docs

# Initialisez le modèle OpenAI
llm = OpenAI(temperature=0.5)


# Fonction pour récupérer les documents et générer une réponse
def ask_question(query):
    relevant_docs = get_relevant_docs(query)
    # Créer une chaîne avec les documents récupérés
    retrieved_info = " ".join([doc[0] for doc in relevant_docs])
    # Utilisez le modèle de langage pour générer une réponse basée sur les documents
    qa = RetrievalQA(llm=llm, retriever=retrieved_info)
    return qa.run(query)
