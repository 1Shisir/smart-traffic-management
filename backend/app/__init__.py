from flask import Flask, redirect, url_for
from flask_socketio import SocketIO, disconnect
from flask_jwt_extended import JWTManager, get_jwt_identity, verify_jwt_in_request
import logging
from app.config import Config
from app.models.user import User, Base as UserBase
from app.models.traffic_data import Base as TrafficBase
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define Session at module level
Session = None

def create_app():
    global Session
    app = Flask(__name__,template_folder='../templates',static_folder='../static')
    app.config.from_object(Config)
    socketio = SocketIO(app, cors_allowed_origins="*")
    jwt = JWTManager(app)

    @socketio.on('connect')
    def handle_connect():
        try:
            verify_jwt_in_request()  # checks JWT from cookies
            user = get_jwt_identity()
            print(f"User {user} connected via Socket.IO")
        except Exception as e:
            print(f"Socket.IO connection rejected: {e}")
            disconnect()

    # Custom unauthorized response: redirect to login
    @jwt.unauthorized_loader
    def unauthorized_callback(error):
        return redirect(url_for('api.login'))

    # Initialize database and create test users
    engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'], echo=app.config['SQLALCHEMY_ECHO'])
    UserBase.metadata.create_all(engine)
    TrafficBase.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Create test users if they don't exist
    for user_data in Config.TEST_USERS:
        if not session.query(User).filter_by(username=user_data['username']).first():
            user = User(username=user_data['username'])
            user.set_password(user_data['password'])
            session.add(user)
    session.commit()

    from app.routes.api import api
    app.register_blueprint(api)

    return app, socketio