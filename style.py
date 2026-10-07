"""ธีมและคอมโพเนนต์ตกแต่งหน้าเว็บ (CSS + HTML) ของบ้านฟ้า"""
import html

import streamlit as st

AVATARS = {"user": "🙂", "assistant": "🏠"}
META = {  # ไอคอน + สเปกย่อของแต่ละสินค้า (สินค้าใหม่ที่ไม่อยู่ในนี้จะใช้ค่าเริ่มต้น)
    "หม้อหุงข้าว RC-500": ("🍚", "1.8 ลิตร · หุงได้ 10 ถ้วย"),
    "หม้อทอดไร้น้ำมัน AF-450": ("🍟", "4.5 ลิตร · 80–200°C"),
    "กาต้มน้ำ KT-170": ("☕", "1.7 ลิตร · 2,200 W"),
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@500;600;700&family=Noto+Sans+Thai:wght@400;500&display=swap');
:root{--ink:#2B2118;--muted:#8A7A6C;--brand:#E8590C;--amber:#F59F00;--sand:#FBEFE0;--line:#F0DFCB}
.stApp{font-family:'Noto Sans Thai',sans-serif}
h1,h2,h3,.hero-title,.pcard b{font-family:'Prompt',sans-serif!important;color:var(--ink)}
#MainMenu,footer{visibility:hidden}
header[data-testid="stHeader"]{background:transparent}
.block-container{max-width:860px;padding-top:1.2rem;padding-bottom:7rem}
.hero{display:flex;gap:1rem;align-items:center;padding:1.3rem 1.5rem;border-radius:22px;color:#fff;
 background:linear-gradient(135deg,#E8590C 0%,#F59F00 100%);box-shadow:0 8px 24px rgba(232,89,12,.28)}
.hero-badge{font-size:2.4rem;background:rgba(255,255,255,.22);border-radius:18px;padding:.4rem .7rem}
.hero-title{font-size:1.9rem;font-weight:700;line-height:1.1;color:#fff!important}
.hero-sub{opacity:.95;font-size:.95rem;margin-top:.2rem}
.chips{display:flex;flex-wrap:wrap;gap:.5rem;margin:.9rem 0 1.2rem}
.chips span{background:#fff;border:1px solid var(--line);border-radius:999px;padding:.25rem .8rem;font-size:.82rem;color:var(--ink)}
.welcome{font-family:'Prompt',sans-serif;font-weight:600;margin:.4rem 0 .6rem;color:var(--ink)}
.pgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:.8rem}
.pcard{background:#fff;border:1px solid var(--line);border-radius:18px;padding:1rem;text-align:center;
 box-shadow:0 2px 10px rgba(120,70,20,.06);transition:.2s}
.pcard:hover{transform:translateY(-3px);box-shadow:0 8px 20px rgba(120,70,20,.14);border-color:var(--amber)}
.pcard .ic{font-size:2.3rem}.pcard b{display:block;margin:.3rem 0 .15rem}.pcard small{color:var(--muted)}
.hint{color:var(--muted);font-size:.88rem;margin-top:.9rem;text-align:center}
[data-testid="stChatMessage"]{background:#fff;border:1px solid var(--line);border-radius:18px;padding:.9rem 1.1rem;
 box-shadow:0 2px 10px rgba(120,70,20,.06);margin-bottom:.6rem}
[data-testid="stChatMessage"]:has([data-testid*="vatarUser"],[data-testid*="Icon-user"]){
 background:linear-gradient(135deg,#FFE8D2,#FFF3E6);border-color:#F8D3B0}
[data-testid="stChatInput"]{border-radius:999px;border:1.5px solid var(--line);box-shadow:0 4px 16px rgba(120,70,20,.1)}
[data-testid="stChatInput"]:focus-within{border-color:var(--brand)}
[data-testid="stSidebar"]{background:var(--sand);border-right:1px solid var(--line)}
[data-testid="stSidebar"] h2{font-size:1.05rem;margin-top:.6rem}
[data-testid="stSidebar"] .stButton>button{border-radius:12px;border:1px solid var(--line);background:#fff;
 text-align:left;justify-content:flex-start;transition:.15s}
[data-testid="stSidebar"] .stButton>button:hover{border-color:var(--brand);color:var(--brand);transform:translateX(3px)}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:14px;background:#FFFDF9}
.src{border-left:4px solid var(--amber);background:#fff;border-radius:10px;padding:.6rem .8rem;margin:.5rem 0}
.src b{font-size:.9rem}.src code{font-size:.75rem}
.src p{margin:.3rem 0 0;color:#5b4d40;font-size:.85rem;line-height:1.5}
.bar{height:6px;background:var(--sand);border-radius:99px;margin:.35rem 0;overflow:hidden}
.bar i{display:block;height:100%;background:linear-gradient(90deg,var(--amber),var(--brand));border-radius:99px}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero():
    st.markdown(
        """<div class="hero"><div class="hero-badge">🏠</div><div>
<div class="hero-title">บ้านฟ้า</div>
<div class="hero-sub">ผู้ช่วยคู่มือเครื่องใช้ไฟฟ้า BaanSuk · ถามได้ทั้งไทยและอังกฤษ</div></div></div>
<div class="chips"><span>📖 ตอบจากคู่มือจริง</span><span>🔎 แสดงแหล่งอ้างอิงทุกคำตอบ</span>
<span>🛡️ ไม่เดา ไม่แต่งข้อมูล</span></div>""",
        unsafe_allow_html=True,
    )


def product_cards(products):
    cards = ""
    for name in products:
        icon, spec = META.get(name, ("🔌", "เครื่องใช้ไฟฟ้า BaanSuk"))
        cards += f'<div class="pcard"><div class="ic">{icon}</div><b>{html.escape(name)}</b><small>{spec}</small></div>'
    st.markdown(
        f'<div class="welcome">👋 สวัสดีครับ ผมช่วยตอบเรื่องสินค้าเหล่านี้ได้</div>'
        f'<div class="pgrid">{cards}</div>'
        f'<div class="hint">เลือกรุ่นที่แถบด้านซ้าย แล้วกดตัวอย่างคำถาม หรือพิมพ์ถามได้เลย</div>',
        unsafe_allow_html=True,
    )


def show_sources(hits):
    with st.expander(f"📚 เอกสารอ้างอิง ({len(hits)} ส่วน)"):
        out = ""
        for h in hits:
            pct = int(max(0, min(1, h["score"])) * 100)
            text = html.escape(h["text"]).replace("\n", "<br>")
            out += (
                f'<div class="src"><b>{html.escape(h["title"])}</b> · <code>{h["file"]}</code>'
                f'<div class="bar"><i style="width:{pct}%"></i></div>'
                f'<small>ความเกี่ยวข้อง {pct}%</small><p>{text}</p></div>'
            )
        st.markdown(out, unsafe_allow_html=True)
