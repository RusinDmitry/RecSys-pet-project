import datetime
from typing import Optional

from pydantic import BaseModel, Field   # импортировали нужные библиотеки


class UserGet(BaseModel):
    id: int
    gender: int
    age: int
    country: str
    city: str
    exp_group: int
    os: str
    source:str

    class Config:
        orm_mode = True


class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True

class Response(BaseModel):
    exp_group: str
    recommendations: list[PostGet]

    class Config:
        orm_mode = True

class FeedGet(BaseModel):
    user_id: int
    post_id: int
    user: UserGet
    post: PostGet
    action: str
    time: datetime.datetime

    class Config:
        orm_mode = True