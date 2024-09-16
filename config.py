import os


class Config:
    SQLALCHEMY_DATABASE_URI = 'postgresql://postgres:alice@localhost/rag_db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
