from database import Base,SessionLocal
from sqlalchemy import Column, Integer, String, Boolean, create_engine, func


class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True)
    gender = Column(Integer)
    age = Column(Integer)
    country = Column(String)
    city = Column(String)
    exp_group = Column(Integer)
    os = Column(String)
    source = Column(String)

def show_county_and_os():
    session = SessionLocal()
    users = session.query(User.country,User.os,func.count('*')) \
        .filter(User.exp_group == 3) \
        .group_by(User.country, User.os).having(func.count('*') > 100)\
        .order_by(func.count('*').desc())\
        .all()
    result = [i for i in users]
    print (result)


if __name__ == '__main__':
    show_county_and_os()