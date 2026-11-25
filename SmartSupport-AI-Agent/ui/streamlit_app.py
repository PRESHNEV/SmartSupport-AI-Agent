import streamlit as st, requests
API = "http://127.0.0.1:8001"
st.set_page_config(page_title="SmartSupport", page_icon="🤖")
st.title("🤖 SmartSupport")
st.caption("Customer-support AI agent demo using sample data.")
if "history" not in st.session_state: st.session_state.history = []
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["text"])
        if m.get("tools"):
            with st.expander("Tool calls and results"): st.json(m["tools"])
q = st.chat_input("Describe your support issue...")
if q:
    st.session_state.history.append({"role":"user","text":q})
    with st.chat_message("user"): st.write(q)
    with st.chat_message("assistant"):
        try:
            r = requests.post(API+"/chat", json={"message":q}, timeout=30); r.raise_for_status()
            d = r.json(); st.markdown(d["reply"]); st.caption("Intent: "+d["intent"])
            with st.expander("Agent tool calls and results"): st.json(d["tool_calls"])
            st.session_state.history.append({"role":"assistant","text":d["reply"],"tools":d["tool_calls"]})
        except requests.RequestException as e: st.error(f"Start API on port 8001 first: {e}")
