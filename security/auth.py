# ---------------------------------------------------------------------------
# auth.py — password gate for the public demo.
#
# The password is read only from the DEMO_PASSWORD environment variable —
# never hardcoded. If it isn't set, access fails closed with a config error
# rather than silently letting everyone in.
# ---------------------------------------------------------------------------

import os

import streamlit as st


def require_auth() -> None:
    """Block the rest of the script from running until the visitor enters
    the correct password. Call this as the first thing after page config.
    """
    if st.session_state.get("_authenticated"):
        return

    password = os.environ.get("DEMO_PASSWORD")

    if not password:
        st.error(
            "This demo is misconfigured: no access password is set on the "
            "server. Set the DEMO_PASSWORD environment variable to enable "
            "access."
        )
        st.stop()

    st.markdown(
        '<div class="pa-gate">'
        '<div class="pa-gate-title">Portfolio Allocation Agent</div>'
        '<div class="pa-gate-subtitle">This is a private demo. '
        'Enter the access password to continue.</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    with st.form("pa_password_gate"):
        entered = st.text_input("Access password", type="password")
        submitted = st.form_submit_button("Unlock")

    if submitted:
        if entered == password:
            st.session_state["_authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password. Please try again.")

    st.stop()
