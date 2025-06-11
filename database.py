from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(100), nullable=False)

class Crop(Base):
    __tablename__ = 'crops'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    image_path = Column(String(100), nullable=False)
    prediction = Column(String(255), nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

engine = create_engine('mysql+mysqlconnector://root:YourPassword@localhost/crop_db')
Base.metadata.create_all(engine)

def add_crop(user_id, image_path, prediction):
    Session = sessionmaker(bind=engine)
    session = Session()
    crop = Crop(user_id=user_id, image_path=image_path, prediction=prediction)
    session.add(crop)
    session.commit()
    session.close()

def get_user_crops(user_id):
    Session = sessionmaker(bind=engine)
    session = Session()
    crops = session.query(Crop).filter_by(user_id=user_id).all()
    session.close()
    return crops

def add_user(username, password):
    Session = sessionmaker(bind=engine)
    session = Session()
    user = User(username=username, password=password)
    session.add(user)
    session.commit()
    user_id = user.id
    session.close()
    return user_id

def check_user(username, password):
    Session = sessionmaker(bind=engine)
    session = Session()
    user = session.query(User).filter_by(username=username, password=password).first()
    session.close()
    return user.id if user else None
