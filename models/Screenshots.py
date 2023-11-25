import datetime
import uuid

from sqlalchemy import ForeignKey
from sqlalchemy.orm import mapped_column, Mapped, relationship

from models.BaseModel import EntityMeta
from models.Video import Video


class Screenshot(EntityMeta):
    __tablename__ = "screenshot"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    path: Mapped[str] = mapped_column(unique=True)
    time: Mapped[float]
    type: Mapped[str]
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("video.id"))
    video: Mapped["Video"] = relationship(back_populates="screenshots")
    created: Mapped[datetime.datetime] = mapped_column(default=datetime.datetime.now)

    def normalize(self):
        return {
            "id": self.id.__str__(),
            "path": self.path.__str__(),
            "time": self.time.__str__(),
            "type": self.type.__str__(),
            "video_id": self.video_id.__str__(),
            "created": self.created.__str__(),
        }