# models.py defines in-memory data structure the indexing layer will use internally while processing
# does not touch database or any logic - just shape definitions

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

# enum restricting chunk_level to exactly parent, child - case sensitive and exact
class ChunkLevel(str, Enum):
    PARENT = "parent"
    CHILD = "child"

# groups all six boolean flags together instead of being scattered across Chunk
# bundled into one object to pass around, check, and reason with - flags.has_table reads more clearly than chunk_has_table
# technically this should always be set before a chunk hits the DB, but None default is 
# fine since chunks are constructed before the document record is written and the ID is assigned in indexer.py
@dataclass 
class RichContentFlags:
    has_table: bool = False
    has_code: bool = False
    has_math: bool = False
    has_definition_list: bool = False
    has_admonition: bool = False
    has_steps: bool = False

# central object, representing one chunk (parent or child) as it flows through pipeline
@dataclass
class Chunk:
    id: UUID = field(default_factory=uuid4)
    document_id: UUID | None = None # indexer.py will set before insert
    parent_id: UUID | None = None
    chunk_level: ChunkLevel = ChunkLevel.CHILD
    chunk_index: int = 0
    section_heading: str | None = None
    chunk_text: str = ""
    html_text: str | None = None
    flags: RichContentFlags = field(default_factory=RichContentFlags)
    char_start: int = 0
    char_end: int = 0
    token_start: int = 0
    token_end: int = 0
    embedding: list[float] | None = None
    source_url: str = ""
    fetched_at: datetime | None = None
    depth: int = 0
    title: str | None = None
