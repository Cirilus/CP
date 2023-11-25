import datetime
import uuid

from pydantic import BaseModel


class Video(BaseModel):
    id: uuid.UUID
    name: str
    path: str
    status: str
    created: datetime.datetime

    class Config:
        from_attributes = True
