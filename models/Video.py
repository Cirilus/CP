import datetime
import uuid
from typing import List

from sqlalchemy.orm import mapped_column, Mapped, relationship

from models.BaseModel import EntityMeta


class Video(EntityMeta):
    __tablename__ = "video"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    name: Mapped[str]
    path: Mapped[str] = mapped_column(unique=True)
    status: Mapped[str]
    screenshots: Mapped[List["Screenshot"]] = relationship(back_populates="video")
    created: Mapped[datetime.datetime] = mapped_column(default=datetime.datetime.now)

    def normalize(self):
        return {
            "id": self.id.__str__(),
            "name": self.name.__str__(),
            "path": self.path.__str__(),
            "status": self.status.__str__(),
            "created": self.created.__str__(),
        }