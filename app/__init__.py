from flask import Flask
from flask_socketio import SocketIO
from app.routes.api import api
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_app():
    app = Flask(__name__)
    socketio = SocketIO(app, cors_allowed_origins="*")
    app.register_blueprint(api)

    return app, socketio