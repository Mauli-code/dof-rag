import datetime

from fastlite import database
from sqlite_vec import load


class Document:
    id: int
    title: str
    url: str
    file_path: str
    created_at: datetime.datetime


class Chunk:
    id: int
    document_id: int
    text: str
    embedding: bytes
    created_at: datetime.datetime


db = database("dof_db/db.sqlite")
db.conn.enable_load_extension(True)
load(db.conn)
db.conn.enable_load_extension(False)

db.create(Document, pk="id")
db.create(Chunk, pk="id", foreign_keys=[("document_id", "document")])
