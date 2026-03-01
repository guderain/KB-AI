from pydantic import BaseModel


class SourceContentResponse(BaseModel):
    path: str
    title: str
    content: str

