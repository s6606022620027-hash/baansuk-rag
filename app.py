import re
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pythainlp.util import normalize
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent / "data"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # เล็ก รองรับไทย/อังกฤษ
# ลองตามลำดับ: ถ้าโมเดลแรกใช้ไม่ได้ (เช่น ถูกถอด) จะข้ามไปตัวถัดไป
LLM_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
CHUNK_SIZE, OVERLAP = 450, 80

SYSTEM_PROMPT = """คุณคือ "น้องสุข" ผู้ช่วยตอบคำถามคู่มือเครื่องใช้ไฟฟ้าแบรนด์ BaanSuk (บ้านสุข)
กฎที่ต้องทำตามอย่างเคร่งครัด:
1. ตอบจาก [บริบท] ที่ให้มาเท่านั้น ห้ามใช้ความรู้ภายนอกหรือเดาเอง
2. ถ้าบริบทไม่มีข้อมูลที่ตอบคำถามได้ ให้ตอบว่า "ไม่พบข้อมูลในคู่มือ" และแนะนำให้โทรศูนย์บริการ ห้ามแต่งคำตอบ
3. ท้ายคำตอบให้อ้างอิงแหล่งที่มา เช่น (อ้างอิง: ชื่อไฟล์) ตามที่ใช้ตอบจริง
4. ตอบเป็นภาษาเดียวกับผู้ใช้ กระชับ เป็นขั้นตอนเมื่อเป็นวิธีใช้งาน
5. ถ้าคำถามไม่ระบุรุ่นสินค้าและคำตอบต่างกันตามรุ่น ให้ถามผู้ใช้ว่าใช้รุ่นไหน"""


def clean_text(text: str) -> str:
    text = text.replace("\u200b", "")
    text = normalize(text)  # จัดสระ/วรรณยุกต์ซ้ำซ้อนของภาษาไทย
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP):
    """แบ่งตามย่อหน้า รวมให้ได้ขนาดใกล้ size; ย่อหน้ายาวมากตัดแบบมี overlap"""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 1 <= size:
            cur = f"{cur}\n{p}".strip()
            continue
        if cur:
            chunks.append(cur)
        if len(p) > size:
            for i in range(0, len(p), size - overlap):
                chunks.append(p[i : i + size])
            cur = ""
        else:
            cur = p
    if cur:
        chunks.append(cur)
    return chunks


@st.cache_resource(show_spinner="กำลังโหลดโมเดลและสร้างดัชนีเอกสาร...")
def build_index():
    model = SentenceTransformer(EMBED_MODEL)
    records = []
    for f in sorted(DATA_DIR.glob("*.txt")):
        raw = clean_text(f.read_text(encoding="utf-8"))
        title = raw.splitlines()[0].lstrip("# ").strip()
        for i, c in enumerate(chunk_text(raw), 1):
            # ใส่ชื่อเอกสารนำหน้า chunk ช่วยให้ค้นแยกรุ่นสินค้าได้แม่นขึ้น
            records.append({"file": f.name, "title": title, "n": i, "text": c})
    emb = model.encode(
        [f"{r['title']}\n{r['text']}" for r in records],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(np.asarray(emb, dtype="float32"))
    return model, index, records


def retrieve(query, model, index, records, k, min_score):
    q = model.encode([query], normalize_embeddings=True)
    scores, ids = index.search(np.asarray(q, dtype="float32"), k)
    hits = [
        {**records[i], "score": float(s)}
        for s, i in zip(scores[0], ids[0])
        if i != -1 and s >= min_score
    ]
    return hits


def ask_llm(question, history, hits):
    context = "\n\n".join(
        f"[{h['file']} #{h['n']}]\n{h['text']}" for h in hits
    )
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs += [{"role": m["role"], "content": m["content"]} for m in history[-6:]]
    msgs.append(
        {"role": "user", "content": f"[บริบท]\n{context}\n\n[คำถาม]\n{question}"}
    )
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    last_err = None
    for model_name in LLM_MODELS:
        try:
            res = client.chat.completions.create(
                model=model_name,
                messages=msgs,
                temperature=0.1,
                max_completion_tokens=2000,  # เผื่อโทเคนส่วนการให้เหตุผลของโมเดล
                extra_body={"reasoning_effort": "low"},
            )
            return res.choices[0].message.content
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise last_err


def show_sources(hits):
    with st.expander(f"📚 เอกสารอ้างอิง ({len(hits)} ส่วน)"):
        for h in hits:
            st.markdown(f"**{h['title']}** · `{h['file']}` · ความเกี่ยวข้อง {h['score']:.2f}")
            st.caption(h["text"])


st.set_page_config(page_title="น้องสุข - ผู้ช่วยคู่มือ BaanSuk", page_icon="🏠")
st.title("🏠 น้องสุข ผู้ช่วยคู่มือเครื่องใช้ไฟฟ้า BaanSuk")
st.caption("ถามวิธีใช้ แก้ปัญหาเบื้องต้น และเงื่อนไขการรับประกัน ตอบจากคู่มือเท่านั้น")

model, index, records = build_index()

with st.sidebar:
    st.header("ตั้งค่า")
    top_k = st.slider("จำนวนส่วนเอกสารที่ค้น (top-k)", 1, 8, 4)
    min_score = st.slider("เกณฑ์ความเกี่ยวข้องขั้นต่ำ", 0.0, 0.8, 0.30, 0.05)
    st.markdown(f"เอกสารในระบบ: **{len({r['file'] for r in records})}** ไฟล์ / **{len(records)}** chunk")
    st.markdown("**ตัวอย่างคำถาม**")
    st.markdown("- หม้อหุงข้าว RC-500 ขึ้น E2 ทำไงดี\n- หม้อทอด AF-450 รับประกันกี่ปี\n- How do I descale the kettle?")
    if st.button("🗑️ ล้างการสนทนา"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("hits"):
            show_sources(m["hits"])

if question := st.chat_input("พิมพ์คำถามเกี่ยวกับสินค้า..."):
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # คำถามสั้น/ต่อเนื่อง (เช่น "แล้วรุ่นนี้ล่ะ") ให้ต่อกับคำถามก่อนหน้าเพื่อค้นหา
    search_q = question
    prev_users = [m["content"] for m in history if m["role"] == "user"]
    if len(question) < 25 and prev_users:
        search_q = f"{prev_users[-1]} {question}"

    with st.chat_message("assistant"):
        hits = retrieve(search_q, model, index, records, top_k, min_score)
        if not hits:
            answer = "ไม่พบข้อมูลในคู่มือ ลองระบุรุ่นสินค้าให้ชัดเจนขึ้น หรือติดต่อศูนย์บริการ BaanSuk โทร 02-555-0199"
        else:
            with st.spinner("กำลังค้นคู่มือและเรียบเรียงคำตอบ..."):
                try:
                    answer = ask_llm(question, history, hits)
                except Exception as e:
                    answer = f"เกิดข้อผิดพลาดในการเรียก LLM: {e}"
        st.markdown(answer)
        if hits:
            show_sources(hits)
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "hits": hits}
    )
