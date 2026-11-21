"""Scores a candidate's answer with an LLM plus local heuristics."""
import re

from .llm import generate

FILLERS = ["um", "uh", "like", "you know", "basically", "actually", "literally", "sort of",
           "kind of", "i mean", "right"]


def filler_count(text: str) -> dict:
    low = " " + text.lower() + " "
    counts = {}
    for f in FILLERS:
        n = len(re.findall(r"(?<![a-z])" + re.escape(f) + r"(?![a-z])", low))
        if n:
            counts[f] = n
    return counts


def score_answer(question: dict, answer: str, jd_text: str) -> dict:
    words = len(answer.split())
    fillers = filler_count(answer)
    prompt = f"""You are a strict but fair interview coach.

QUESTION ({question.get('type')}): {question['question']}
JOB DESCRIPTION (excerpt): {jd_text[:1500]}
CANDIDATE ANSWER: {answer}

NOTE: This answer may come from automatic speech-to-text, so it can contain misheard words (e.g. 'reddisk' for Redis). Do NOT penalise spelling, typos or transcription errors; judge the intended meaning.

Score the answer from 1 to 10 on:
- clarity: structure and easy to follow
- relevance: answers what was asked and fits the job
- depth: technical or experiential detail, concrete examples, numbers
- star: for behavioral answers, how well it covers Situation, Task, Action, Result (for other types, judge logical structure)

Return ONLY JSON:
{{"clarity": int, "relevance": int, "depth": int, "star": int,
  "strengths": "one sentence", "improvements": "one or two sentences",
  "model_answer_hint": "2-3 sentence outline of a stronger answer"}}"""
    try:
        s = generate(prompt, json_mode=True)
        for k in ("clarity", "relevance", "depth", "star"):
            s[k] = max(1, min(10, int(s.get(k, 5))))
    except Exception as e:  # never crash the interview because of one bad LLM reply
        s = {"clarity": 5, "relevance": 5, "depth": 5, "star": 5,
             "strengths": "Scoring service unavailable.",
             "improvements": f"Could not score this answer ({e}).",
             "model_answer_hint": ""}
    s["overall"] = round((s["clarity"] + s["relevance"] + s["depth"] + s["star"]) / 4, 1)
    s["word_count"] = words
    s["fillers"] = fillers
    s["filler_total"] = sum(fillers.values())
    return s
