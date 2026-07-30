# ---------------------------------------------------------------------------
# identity.py — resolves a per-visitor identifier for rate limiting.
#
# Streamlit exposes the client IP via st.context.ip_address (added in
# recent Streamlit versions). If it isn't available in this deployment
# environment, fall back to a per-browser-session UUID stored in
# session_state, per the fallback the user explicitly approved.
# ---------------------------------------------------------------------------

import uuid

import streamlit as st


def get_identifier() -> str:
    ip = None
    try:
        ip = st.context.ip_address
    except Exception:
        ip = None

    if ip:
        return f"ip:{ip}"

    if "_session_identifier" not in st.session_state:
        st.session_state["_session_identifier"] = str(uuid.uuid4())

    return f"session:{st.session_state['_session_identifier']}"
