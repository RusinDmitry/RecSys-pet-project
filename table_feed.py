from database import Base,SessionLocal
from sqlalchemy import Column, Integer, String, Boolean,TIMESTAMP, ForeignKey, func
from sqlalchemy.orm import relationship
from table_post import Post
from table_user import User



class Feed(Base):
    __tablename__ = "feed_action"
    user_id = Column(Integer, ForeignKey(User.id), primary_key=True)
    user = relationship("User")
    post_id = Column(Integer, ForeignKey(Post.id), primary_key=True)
    post = relationship("Post")
    action = Column(String)
    time = Column(TIMESTAMP)
