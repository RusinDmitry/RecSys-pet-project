from database import Base,SessionLocal
from sqlalchemy import Column, Integer, String, Boolean, create_engine


class Post(Base):
    __tablename__ = "post"
    id = Column(Integer, primary_key=True)
    text = Column(String)
    topic = Column(String)

def easy_select():
    session = SessionLocal()
    posts = session.query(Post.id) \
        .filter(Post.topic == "business")\
        .order_by(Post.id.desc())\
        .limit(10) \
        .all()
    result = [x[0] for x in posts]
    print(result)

if __name__ == '__main__':
    easy_select()