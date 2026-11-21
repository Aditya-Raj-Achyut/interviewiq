"""Generates tailored interview questions from resume + job description."""
from .llm import generate


def generate_questions(resume_ctx: str, jd_text: str, role: str, n: int, difficulty: str,
                       missing_skills: list) -> list:
    n_tech = max(1, round(n * 0.5))
    n_beh = max(1, round(n * 0.3))
    n_hr = max(1, n - n_tech - n_beh)
    prompt = f"""You are a senior interviewer hiring for the role: {role}.
Difficulty: {difficulty}.

JOB DESCRIPTION:
{jd_text[:3000]}

RELEVANT PARTS OF THE CANDIDATE'S RESUME:
{resume_ctx[:3000]}

Skills required by the JD but missing from the resume: {', '.join(missing_skills) or 'none'}

Create exactly {n} interview questions:
- {n_tech} "technical" questions (reference the candidate's real projects/skills; include 1 probing a missing skill if any)
- {n_beh} "behavioral" questions answerable with the STAR method
- {n_hr} "hr" questions (motivation, culture fit, career goals)

Return ONLY a JSON array. Each item: {{"type": "technical|behavioral|hr", "question": "...", "skill": "main skill tested"}}"""
    data = generate(prompt, json_mode=True)
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), [])
    questions = [q for q in data if isinstance(q, dict) and q.get("question")]
    return questions[:n]
