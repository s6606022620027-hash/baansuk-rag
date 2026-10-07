import re
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pythainlp.util import normalize
from sentence_transformers import SentenceTransformer

from style import AVATARS, hero, inject_css, product_cards, show_sources

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

# รายการสินค้า: ชื่อที่แสดง -> ไฟล์เอกสารของสินค้านั้น + ตัวอย่างคำถาม
# เพิ่มสินค้าใหม่ได้ที่นี่ (ไฟล์ใน data/ ที่ไม่อยู่ในรายการ ถือเป็นเอกสารทั่วไป เช่น รับประกัน ศูนย์บริการ)
PRODUCTS = {
    "หม้อหุงข้าว RC-500": {
        "files": ["01_rice_cooker_rc500.txt"],
        "questions": [
            "หุงข้าวขาว 4 ถ้วยใช้เวลากี่นาที",
            "ขึ้น E2 หมายถึงอะไร ต้องทำยังไง",
            "โหมดอุ่นอุ่นข้าวได้นานแค่ไหน",
            "ล้างหม้อในยังไงไม่ให้เคลือบลอก",
        ],
    },
    "หม้อทอดไร้น้ำมัน AF-450": {
        "files": ["02_air_fryer_af450.txt"],
        "questions": [
            "ทอดปีกไก่ 500 กรัมใช้กี่องศากี่นาที",
            "ใช้งานครั้งแรกต้องทำอะไรก่อน",
            "มีควันขาวเยอะ เกิดจากอะไร",
            "ขึ้น E5 ต้องทำอย่างไร",
        ],
    },
    "กาต้มน้ำ KT-170": {
        "files": ["03_kettle_kt170.txt"],
        "questions": [
            "How do I descale the kettle?",
            "ต้มน้ำ 1 ลิตรใช้เวลากี่นาที",
            "กาตัดไฟเองบ่อย ต้องเช็คอะไร",
            "Keep warm ทำงานกี่องศา นานแค่ไหน",
        ],
    },
}
ALL_MODELS = "ทุกรุ่น (ให้ระบบค้นเอง)"


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


def retrieve(query, model, index, records, k, min_score, allowed_files=None):
    q = model.encode([query], normalize_embeddings=True)
    scores, ids = index.search(np.asarray(q, dtype="float32"), index.ntotal)
    hits = []
    for s, i in zip(scores[0], ids[0]):
        if i == -1 or s < min_score:
            continue
        if allowed_files is not None and records[i]["file"] not in allowed_files:
            continue
        hits.append({**records[i], "score": float(s)})
        if len(hits) == k:
            break
    return hits


def ask_llm(question, history, hits, product=None):
    context = "\n\n".join(
        f"[{h['file']} #{h['n']}]\n{h['text']}" for h in hits
    )
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs += [{"role": m["role"], "content": m["content"]} for m in history[-6:]]
    note = f"(ผู้ใช้กำลังถามเกี่ยวกับสินค้า: {product})\n" if product else ""
    msgs.append(
        {"role": "user", "content": f"[บริบท]\n{context}\n\n[คำถาม]\n{note}{question}"}
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


st.set_page_config(page_title="น้องสุข - ผู้ช่วยคู่มือ BaanSuk", page_icon="🏠")
inject_css()
hero()

model, index, records = build_index()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("1) เลือกสินค้า")
    choice = st.selectbox("คุณอยากถามเรื่องสินค้ารุ่นไหน?", [ALL_MODELS] + list(PRODUCTS))

    st.header("2) กดถามได้เลย")
    if choice == ALL_MODELS:
        samples = [v["questions"][0] for v in PRODUCTS.values()]
        st.caption("ตัวอย่างคำถาม (เลือกสินค้าด้านบนเพื่อดูคำถามเฉพาะรุ่น)")
    else:
        samples = PRODUCTS[choice]["questions"]
        st.caption(f"ตัวอย่างคำถามของ {choice}")
    for i, q in enumerate(samples):
        if st.button(q, key=f"sample_{i}", use_container_width=True):
            st.session_state.queued = q

    with st.expander("⚙️ ตั้งค่าเพิ่มเติม"):
        detail = st.select_slider(
            "ความละเอียดของคำตอบ",
            options=["สั้นกระชับ", "ปกติ", "ละเอียดมาก"],
            value="ปกติ",
            help="ยิ่งละเอียด ระบบยิ่งอ่านคู่มือหลายส่วนมาตอบ แต่อาจช้าลงเล็กน้อย",
        )
        strict = st.select_slider(
            "ความเข้มงวดในการหาข้อมูล",
            options=["ผ่อนปรน", "ปกติ", "เข้มงวด"],
            value="ปกติ",
            help="ถ้าตอบว่า 'ไม่พบข้อมูล' บ่อยเกินไป ให้เลือก 'ผ่อนปรน' / "
            "ถ้าคำตอบไม่ตรงเรื่อง ให้เลือก 'เข้มงวด'",
        )
    top_k = {"สั้นกระชับ": 3, "ปกติ": 4, "ละเอียดมาก": 6}[detail]
    min_score = {"ผ่อนปรน": 0.20, "ปกติ": 0.30, "เข้มงวด": 0.45}[strict]

    if st.button("🗑️ ล้างการสนทนา", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption(f"เอกสารในระบบ: {len({r['file'] for r in records})} ไฟล์ / {len(records)} ส่วน")

if choice == ALL_MODELS:
    allowed = None
else:
    product_files = {f for v in PRODUCTS.values() for f in v["files"]}
    general = {r["file"] for r in records} - product_files  # เอกสารทั่วไป ใช้ได้กับทุกรุ่น
    allowed = set(PRODUCTS[choice]["files"]) | general

welcome = st.empty()  # พื้นที่การ์ดต้อนรับ ซ่อนทันทีเมื่อเริ่มแชต
if not st.session_state.messages:
    with welcome.container():
        product_cards(PRODUCTS)

for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar=AVATARS[m["role"]]):
        st.markdown(m["content"])
        if m.get("hits"):
            show_sources(m["hits"])

placeholder = "พิมพ์คำถามเกี่ยวกับสินค้า..." if choice == ALL_MODELS else f"ถามเรื่อง {choice}..."
question = st.chat_input(placeholder)
if "queued" in st.session_state:
    question = st.session_state.pop("queued")

if question:
    welcome.empty()
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(question)

    # คำถามสั้น/ต่อเนื่อง (เช่น "แล้วรุ่นนี้ล่ะ") ให้ต่อกับคำถามก่อนหน้าเพื่อค้นหา
    search_q = question
    prev_users = [m["content"] for m in history if m["role"] == "user"]
    if len(question) < 25 and prev_users:
        search_q = f"{prev_users[-1]} {question}"
    if choice != ALL_MODELS:
        search_q = f"{choice} {search_q}"

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        hits = retrieve(search_q, model, index, records, top_k, min_score, allowed)
        if not hits:
            answer = "ไม่พบข้อมูลในคู่มือ ลองเลือกรุ่นสินค้าที่แถบด้านซ้าย หรือติดต่อศูนย์บริการ BaanSuk โทร 02-555-0199"
        else:
            with st.spinner("กำลังค้นคู่มือและเรียบเรียงคำตอบ..."):
                try:
                    answer = ask_llm(
                        question, history, hits, None if choice == ALL_MODELS else choice
                    )
                except Exception as e:
                    answer = f"เกิดข้อผิดพลาดในการเรียก LLM: {e}"
        st.markdown(answer)
        if hits:
            show_sources(hits)
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "hits": hits}
    )
