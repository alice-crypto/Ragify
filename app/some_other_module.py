from app.models import Document
from app.db import db


def get_relevant_docs(query):
    relevant_docs = Document.query.filter(Document.content.ilike(f'%{query}%')).all()
    return relevant_docs


def add_document(content):
    new_doc = Document(content=content)
    db.session.add(new_doc)
    db.session.commit()
