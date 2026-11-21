import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.llm import extract_json
from modules.resume_parser import chunk_text, find_skills, skill_gap
from modules.scorer import filler_count


def test_extract_json():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here: [1, 2]') == [1, 2]


def test_skills_and_gap():
    g = skill_gap("I know Python, SQL and Docker", "Need Python, SQL, Kubernetes")
    assert g["matched"] == ["python", "sql"] and g["missing"] == ["kubernetes"]
    assert "c++" in find_skills("Experienced in C++ and Git")


def test_fillers_and_chunks():
    assert filler_count("Um, I basically built it, you know")["um"] == 1
    assert len(chunk_text("word " * 500, 100, 20)) > 3


if __name__ == "__main__":
    test_extract_json(); test_skills_and_gap(); test_fillers_and_chunks()
    print("all tests passed")
