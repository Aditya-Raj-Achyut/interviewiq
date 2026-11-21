import streamlit as st

from modules import analytics
from modules.question_gen import generate_questions
from modules.resume_parser import ResumeIndex, extract_text, skill_gap
from modules.scorer import score_answer

st.set_page_config(page_title="InterviewIQ", page_icon="🎤", layout="wide")

DEFAULTS = {"stage": "setup", "questions": [], "idx": 0, "results": [], "gap": None, "role": ""}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def reset():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v if not isinstance(v, (list, dict)) else type(v)()


st.title("🎤 InterviewIQ")
st.caption("AI mock interviewer: tailored questions from your resume, voice answers, instant scoring.")

with st.sidebar:
    st.header("Settings")
    n_q = st.slider("Number of questions", 3, 10, 5)
    difficulty = st.selectbox("Difficulty", ["Entry level", "Mid level", "Senior"], index=1)
    st.divider()
    if st.button("🔄 Start over"):
        reset()
        st.rerun()

# ---------------------------------------------------------------- SETUP
if st.session_state.stage == "setup":
    c1, c2 = st.columns(2)
    with c1:
        pdf = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
        role = st.text_input("Target role", placeholder="e.g. AI/ML Engineer")
    with c2:
        jd = st.text_area("Paste the job description", height=230)

    if st.button("🚀 Analyze & start interview", type="primary"):
        if not (pdf and jd.strip() and role.strip()):
            st.warning("Please add a resume, a role and a job description.")
        else:
            try:
                with st.spinner("Reading resume and building your interview..."):
                    resume_text = extract_text(pdf)
                    if len(resume_text) < 50:
                        st.error("Could not read text from this PDF (is it a scanned image?).")
                        st.stop()
                    gap = skill_gap(resume_text, jd)
                    ctx = "\n".join(ResumeIndex(resume_text).search(jd[:1500], k=4))
                    qs = generate_questions(ctx, jd, role, n_q, difficulty, gap["missing"])
                if not qs:
                    st.error("The AI returned no questions. Try again.")
                    st.stop()
                st.session_state.update(stage="interview", questions=qs, gap=gap, role=role,
                                        jd=jd, idx=0, results=[])
                st.rerun()
            except Exception as e:
                st.error(f"Something went wrong: {e}")

# ------------------------------------------------------------ INTERVIEW
elif st.session_state.stage == "interview":
    qs, i = st.session_state.questions, st.session_state.idx
    gap = st.session_state.gap

    with st.expander("📊 Resume vs job description", expanded=(i == 0)):
        st.metric("Skill match", f"{gap['match_pct']}%")
        st.write("✅ **Matched:**", ", ".join(gap["matched"]) or "-")
        st.write("⚠️ **Missing from resume:**", ", ".join(gap["missing"]) or "-")

    st.progress(i / len(qs), text=f"Question {i + 1} of {len(qs)}")
    q = qs[i]
    st.subheader(f"{q.get('type', '').title()} question")
    st.info(q["question"])

    ans_key = f"ans_{i}"
    if hasattr(st, "audio_input"):
        audio = st.audio_input("🎙️ Record your answer (optional)", key=f"aud_{i}")
        if audio is not None and st.button("📝 Transcribe recording", key=f"tr_{i}"):
            try:
                from modules.speech import transcribe

                with st.spinner("Transcribing with Whisper (first run downloads the model)..."):
                    st.session_state[ans_key] = transcribe(audio.getvalue())
            except Exception as e:
                st.error(f"Transcription failed: {e}")

    answer = st.text_area("Your answer (edit the transcript or just type)", key=ans_key, height=180)

    if st.button("✅ Submit answer", type="primary"):
        if len(answer.split()) < 5:
            st.warning("Please give a fuller answer (at least a few words).")
        else:
            with st.spinner("Scoring your answer..."):
                score = score_answer(q, answer, st.session_state.jd)
            st.session_state.results.append({"question": q, "answer": answer, "score": score})
            if i + 1 >= len(qs):
                st.session_state.stage = "report"
            else:
                st.session_state.idx += 1
            st.rerun()

# --------------------------------------------------------------- REPORT
else:
    results, gap = st.session_state.results, st.session_state.gap
    st.header("📈 Your interview report")
    avg = analytics.overall_average(results)
    m1, m2, m3 = st.columns(3)
    m1.metric("Overall score", f"{avg} / 10")
    m2.metric("Skill match", f"{gap['match_pct']}%")
    m3.metric("Total filler words", sum(r["score"]["filler_total"] for r in results))

    c1, c2 = st.columns(2)
    c1.plotly_chart(analytics.score_trend(results), use_container_width=True)
    c2.plotly_chart(analytics.radar(results), use_container_width=True)
    c3, c4 = st.columns(2)
    c3.plotly_chart(analytics.filler_chart(results), use_container_width=True)
    c4.plotly_chart(analytics.skill_gap_chart(gap), use_container_width=True)

    st.subheader("Question-by-question feedback")
    for n, r in enumerate(results, 1):
        s = r["score"]
        with st.expander(f"Q{n}: {r['question']['question']}  —  {s['overall']}/10"):
            st.write("**Your answer:**", r["answer"])
            st.write(f"Clarity {s['clarity']} · Relevance {s['relevance']} · Depth {s['depth']} · STAR {s['star']}")
            st.success(f"**Strengths:** {s['strengths']}")
            st.warning(f"**Improve:** {s['improvements']}")
            if s.get("model_answer_hint"):
                st.info(f"**Stronger answer:** {s['model_answer_hint']}")
            if s["fillers"]:
                st.caption("Filler words: " + ", ".join(f"{k} ×{v}" for k, v in s["fillers"].items()))

    st.download_button("⬇️ Download PDF report", analytics.build_pdf(st.session_state.role, gap, results),
                       file_name="interviewiq_report.pdf", mime="application/pdf")
