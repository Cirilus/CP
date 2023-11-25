from typing import List, Optional
from schemas.Screenshot import Screenshot
from schemas.Video import Video


class VideoSchema(Video):
    screenshots: List[Screenshot]

class VideoSchemaList(Video):
    screenshots: List[Screenshot]
    popular: Optional[str] = None