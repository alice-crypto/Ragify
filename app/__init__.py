from flask import Flask
from app.db import db
from app.models import Document
from app.routes import bp as main_bp


def create_app():
    app = Flask(__name__)
    app.config.from_object('config.Config')

    db.init_app(app)

    with app.app_context():
        db.create_all()  # Crée les tables dans la base de données si elles n'existent pas

    app.register_blueprint(main_bp)  # Enregistre le Blueprint

    return app