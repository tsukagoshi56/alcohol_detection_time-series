"""Build and load a lightweight index for face_data_vas."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional


FILE_RE = re.compile(
    r"^subj(?P<sid>\d+)_"
    r"(?P<label>[^_]+)_"
    r"t(?P<time>\d+)s_"
    r"f(?P<frame>\d+)"
    r"(?:_vas(?P<vas>\d+)_class(?P<class>\d+))?"
    r"\.(?P<ext>png|jpg|jpeg)$",
    re.IGNORECASE,
)

VAS_LABEL_RE = re.compile(r"^vas(?P<min>\d+)min$", re.IGNORECASE)


@dataclass
class IndexStats:
    sessions: int
    images: int
    skipped: int


@dataclass
class IndexRecord:
    session_id: str
    subject_id: int
    label: str
    label_type: str
    vas_value: Optional[int]
    vas_time_min: Optional[int]
    time_sec: int
    path: str


def _parse_file(path: Path, session_id: str) -> Optional[IndexRecord]:
    name = path.name
    match = FILE_RE.match(name)
    if not match:
        return None

    sid = int(match.group("sid"))
    label = match.group("label")
    time_sec = int(match.group("time"))
    vas_raw = match.group("vas")
    vas_value = int(vas_raw) if vas_raw is not None else None

    label_type = "normal" if label == "normal" else "vas"

    vas_time_min = None
    vas_match = VAS_LABEL_RE.match(label)
    if vas_match:
        vas_time_min = int(vas_match.group("min"))

    return IndexRecord(
        session_id=session_id,
        subject_id=sid,
        label=label,
        label_type=label_type,
        vas_value=vas_value,
        vas_time_min=vas_time_min,
        time_sec=time_sec,
        path=str(path),
    )


def iter_records(data_root: str) -> Iterator[IndexRecord]:
    root = Path(data_root)
    for session_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        session_id = session_dir.name
        for img_path in sorted(session_dir.iterdir()):
            if not img_path.is_file():
                continue
            record = _parse_file(img_path, session_id)
            if record is None:
                continue
            yield record


def build_index(data_root: str, index_path: str) -> IndexStats:
    index_file = Path(index_path)
    index_file.parent.mkdir(parents=True, exist_ok=True)

    sessions = set()
    images = 0
    skipped = 0

    with index_file.open("w", encoding="utf-8") as f:
        for record in iter_records(data_root):
            sessions.add(record.session_id)
            images += 1
            f.write(json.dumps(record.__dict__, ensure_ascii=True) + "\n")

    # Count skipped files by scanning again and comparing matches
    for session_dir in Path(data_root).iterdir():
        if not session_dir.is_dir():
            continue
        for img_path in session_dir.iterdir():
            if not img_path.is_file():
                continue
            if _parse_file(img_path, session_dir.name) is None:
                skipped += 1

    return IndexStats(sessions=len(sessions), images=images, skipped=skipped)


def load_index(index_path: str) -> Iterable[IndexRecord]:
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            yield IndexRecord(**data)
