from __future__ import annotations

import time
from typing import Any, Dict, List

import pandas as pd
import requests
import streamlit as st

# -----------------------------
# CONFIG
# -----------------------------
API_BASE = "http://127.0.0.1:8000"
HEALTH_ENDPOINT = "/api/health"
TEXT2SQL_ENDPOINT = "/api/text2sql"

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(
    page_title="GARV Text2SQL",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# CSS
# -----------------------------
st.markdown(
    """
<style>
.block-container { padding-top: 1rem !important; max-width: 1200px; }
header[data-testid="stHeader"] { display: none; }
section[data-testid="stSidebar"] { border-right: 1px solid #e5e7eb; }
.chat-container { max-width: 950px; margin: auto; }
.small-muted { color: #6b7280; font-size: 12px; }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# HEALTH CHECK
# -----------------------------
def is_api_alive() -> bool:
    try:
        r = requests.get(API_BASE + HEALTH_ENDPOINT, timeout=1)
        return r.status_code == 200
    except Exception:
        return False


# -----------------------------
# SESSION STATE
# -----------------------------
if "chats" not in st.session_state:
    st.session_state.chats = []
if "active_chat" not in st.session_state:
    st.session_state.active_chat = None


def new_chat() -> None:
    cid = f"chat-{int(time.time() * 1000)}"
    st.session_state.chats.insert(
        0,
        {
            "id": cid,
            "title": "New chat",
            "messages": [
                {
                    "role": "assistant",
                    "type": "text",
                    "content": "Hi! Ask me an airport ops question and I’ll generate SQL + results.",
                }
            ],
            # multi-turn memory
            "chat_history": [],
            "memory_entities": {},
        },
    )
    st.session_state.active_chat = cid


def get_chat() -> Dict[str, Any]:
    if not st.session_state.active_chat:
        new_chat()
    for c in st.session_state.chats:
        if c["id"] == st.session_state.active_chat:
            c.setdefault("chat_history", [])
            c.setdefault("memory_entities", {})
            c.setdefault("messages", [])
            return c
    new_chat()
    return get_chat()


# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.title("GARV Text2SQL")
    st.caption("Enterprise conversational SQL for airport operations")

    if st.button("+ New chat", use_container_width=True, key="btn_new_chat"):
        new_chat()
        st.rerun()

    st.divider()

    for c in st.session_state.chats:
        if st.button(c["title"], use_container_width=True, key=f"chat_select_{c['id']}"):
            st.session_state.active_chat = c["id"]
            st.rerun()

    st.divider()
    connected = is_api_alive()
    st.caption("Status: 🟢 Connected" if connected else "Status: 🔴 API Offline")
    st.caption("Tip: Keep FastAPI running while using this UI.")


# -----------------------------
# RENDER HELPERS
# -----------------------------
def render_assistant_payload(payload: Dict[str, Any], unique_key_prefix: str) -> None:
    """
    payload structure:
      {
        "summary": str,
        "sql": str,
        "df": pd.DataFrame,
        "row_count": int,
      }
    """
    summary = payload.get("summary") or ""
    sql = payload.get("sql") or ""
    df = payload.get("df")

    if summary:
        st.markdown(summary)

    if sql:
        with st.expander("SQL", expanded=False):
            st.code(sql, language="sql")

    if isinstance(df, pd.DataFrame):
        if not df.empty:
            st.markdown(f"<div class='small-muted'>Rows: {len(df)}</div>", unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True, height=280)

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                data=csv_bytes,
                file_name=f"{unique_key_prefix}.csv",
                mime="text/csv",
                key=f"dl_{unique_key_prefix}",
            )
        else:
            st.info("No rows returned.")


# -----------------------------
# MAIN UI
# -----------------------------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

chat = get_chat()

# render conversation (text + structured answer blocks)
for idx, m in enumerate(chat["messages"]):
    role = m.get("role", "assistant")
    mtype = m.get("type", "text")

    with st.chat_message(role):
        if mtype == "text":
            st.markdown(m.get("content", ""))
        elif mtype == "answer":
            # full assistant answer (summary + sql + df + download)
            render_assistant_payload(m.get("payload", {}), unique_key_prefix=f"{chat['id']}_msg{idx}")

prompt = st.chat_input("Ask a question… e.g. Top 5 airports by avg security wait time last 7 days")

if prompt:
    # update title
    if chat["title"] == "New chat":
        chat["title"] = prompt[:30] + ("…" if len(prompt) > 30 else "")

    # append user message
    chat["messages"].append({"role": "user", "type": "text", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # assistant response
    with st.chat_message("assistant"):
        if not is_api_alive():
            st.markdown("❌ API is Offline. Start FastAPI first, then retry.")
            chat["messages"].append(
                {"role": "assistant", "type": "text", "content": "❌ API is Offline. Start FastAPI first, then retry."}
            )
        else:
            with st.spinner("Generating SQL and results…"):
                try:
                    payload = {
                        "question": prompt,
                        "top_k_schema": 5,
                        "return_rows": 50,
                        "enable_viz": False,
                        "chat_history": chat.get("chat_history", []),
                        "memory_entities": chat.get("memory_entities", {}),
                    }

                    r = requests.post(API_BASE + TEXT2SQL_ENDPOINT, json=payload, timeout=120)
                    out = r.json()

                    # update multi-turn memory
                    chat["chat_history"] = out.get("chat_history", chat.get("chat_history", []))
                    chat["memory_entities"] = out.get("memory_entities", chat.get("memory_entities", {}))

                    if out.get("ok"):
                        summary = out.get("explanation", {}).get("summary", "")
                        sql = out.get("final_sql", "")

                        rows = out.get("rows", []) or []
                        df = pd.DataFrame(rows) if rows else pd.DataFrame()

                        assistant_payload = {
                            "summary": summary,
                            "sql": sql,
                            "df": df,
                            "row_count": out.get("row_count"),
                        }

                        # show immediately
                        render_assistant_payload(assistant_payload, unique_key_prefix=f"{chat['id']}_latest")

                        # store persistently
                        chat["messages"].append(
                            {"role": "assistant", "type": "answer", "payload": assistant_payload}
                        )

                    else:
                        stage = out.get("stage", "")
                        msg = out.get("message", "Unknown error")
                        if stage == "clarification":
                            err = f"🤔 {msg}"
                        else:
                            err = f"❌ {msg}"
                        st.markdown(err)
                        chat["messages"].append({"role": "assistant", "type": "text", "content": err})

                except Exception as e:
                    err = f"❌ API error: {e}"
                    st.markdown(err)
                    chat["messages"].append({"role": "assistant", "type": "text", "content": err})

    # rerun so everything renders cleanly from stored messages
    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)