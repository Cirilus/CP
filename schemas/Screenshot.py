import datetime
import uuid

from pydantic import BaseModel


class Screenshot(BaseModel):
    id: uuid.UUID
    path: str
    time: float
    type: str
    created: datetime.datetime

    class Config:
        from_attributes = True
