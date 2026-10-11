import os
import sys
from pathlib import Path

os.environ.setdefault("PLANNING_START", "2026-10-12")
os.environ["MOCK_LLM"] = "true"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from eval.datasets import DATASETS, load_into  # noqa: E402
from src.database import get_connection  # noqa: E402


@pytest.fixture
def make_db(tmp_path):
    """make_db('cheapest_late') -> connection loaded with that tiny dataset."""
    def _make(name: str, **kw):
        conn = get_connection(str(tmp_path / f"{name}.db"))
        load_into(conn, DATASETS[name](**kw))
        return conn
    return _make
