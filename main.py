import threading
import logging
from app import create_app
from app.utils.video_processor import process_video
from app.models.traffic_data import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.config import Config

if __name__ == '__main__':
    app, socketio = create_app()
    engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, echo=Config.SQLALCHEMY_ECHO)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    threading.Thread(target=process_video, args=(app, socketio, session), daemon=True).start()
    logging.info("Starting Flask server")
    socketio.run(app, debug=True, use_reloader=False)