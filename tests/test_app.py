# ---------------------------------------------------------------------------
# UI-level tests for frontend/app.py using Streamlit's AppTest framework.
#
# Covers: the password gate (must still block/unlock correctly after this
# rewrite — the whole point of reading app.py fully before touching it),
# and the Build Portfolio tab's weight validation and Clear Portfolio
# reset. Upload-tab validation is covered separately in
# tests/test_portfolio_upload.py, since AppTest has no way to simulate a
# file_uploader interaction (see that file's module docstring).
#
# None of these tests click "Analyze Portfolio" — doing so would call
# agent.graph, which needs a real GOOGLE_API_KEY and makes real network
# calls (yfinance, Gemini) and would burn real rate-limit quota and write
# to the real usage log. Button *disabled state* and the validation
# messages that drive it are fully testable without ever starting a run.
# ---------------------------------------------------------------------------

import pytest
from streamlit.testing.v1 import AppTest

AAPL_LABEL = "AAPL — Apple Inc."
NVDA_LABEL = "NVDA — NVIDIA Corporation"


@pytest.fixture(autouse=True)
def demo_password(monkeypatch):
    # Set explicitly rather than relying on a local .env file, so these
    # tests are self-contained and pass in a clean CI environment too.
    monkeypatch.setenv("DEMO_PASSWORD", "Portfolio138")


def _fresh_app() -> AppTest:
    at = AppTest.from_file("frontend/app.py")
    at.run()
    return at


def _unlocked_app() -> AppTest:
    at = _fresh_app()
    at.text_input[0].set_value("Portfolio138").run()
    at.button[0].click().run()
    assert not at.exception
    return at


def _button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


class TestPasswordGate:
    def test_blocks_access_without_password(self):
        at = _fresh_app()
        assert not at.exception
        assert any(b.label == "Unlock" for b in at.button)
        # The gated UI (tabs, multiselect) must not have rendered at all.
        assert len(at.multiselect) == 0
        assert len(at.tabs) == 0

    def test_wrong_password_shows_error_and_stays_locked(self):
        at = _fresh_app()
        at.text_input[0].set_value("wrong-password").run()
        at.button[0].click().run()

        assert not at.exception
        assert any("Incorrect password" in e.value for e in at.error)
        assert len(at.multiselect) == 0

    def test_correct_password_unlocks_app(self):
        at = _unlocked_app()
        assert len(at.multiselect) == 1
        assert len(at.tabs) == 2
        assert any(b.label == "Analyze Portfolio" for b in at.button)

    def test_fails_closed_when_demo_password_unset(self, monkeypatch):
        monkeypatch.delenv("DEMO_PASSWORD", raising=False)
        # app.py calls load_dotenv() itself, which would otherwise refill
        # DEMO_PASSWORD from this repo's local .env (present for manual dev
        # testing) and mask the "truly unset" scenario this test targets.
        monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
        at = AppTest.from_file("frontend/app.py")
        at.run()

        assert not at.exception
        assert any("misconfigured" in e.value for e in at.error)
        # No password field at all — nothing to unlock with.
        assert len(at.text_input) == 0


class TestBuildPortfolioWeights:
    def test_weights_summing_to_100_enable_analyze_button(self):
        at = _unlocked_app()
        at.multiselect[0].set_value([AAPL_LABEL, NVDA_LABEL]).run()
        assert len(at.number_input) == 2

        at.number_input[0].set_value(60.0).run()
        at.number_input[1].set_value(40.0).run()

        assert _button(at, "Analyze Portfolio").disabled is False
        assert any("Total: 100.0%" in s.value for s in at.success)

    def test_weights_under_100_keep_button_disabled_and_show_remaining(self):
        at = _unlocked_app()
        at.multiselect[0].set_value([AAPL_LABEL, NVDA_LABEL]).run()
        at.number_input[0].set_value(60.0).run()
        at.number_input[1].set_value(30.0).run()

        assert _button(at, "Analyze Portfolio").disabled is True
        assert any("10.0% remaining" in w.value for w in at.warning)

    def test_weights_over_100_keep_button_disabled_and_show_over_amount(self):
        at = _unlocked_app()
        at.multiselect[0].set_value([AAPL_LABEL, NVDA_LABEL]).run()
        at.number_input[0].set_value(70.0).run()
        at.number_input[1].set_value(60.0).run()

        assert _button(at, "Analyze Portfolio").disabled is True
        assert any("30.0% over 100%" in w.value for w in at.warning)

    def test_no_stocks_selected_keeps_button_disabled(self):
        at = _unlocked_app()
        assert _button(at, "Analyze Portfolio").disabled is True


class TestClearPortfolio:
    def test_clear_resets_multiselect_weights_and_session_state(self):
        at = _unlocked_app()
        at.multiselect[0].set_value([AAPL_LABEL, NVDA_LABEL]).run()
        at.number_input[0].set_value(60.0).run()
        at.number_input[1].set_value(40.0).run()
        assert _button(at, "Analyze Portfolio").disabled is False

        version_before = at.session_state["portfolio_version"]
        _button(at, "Clear Portfolio").click().run()

        # Widgets remounted empty.
        assert at.multiselect[0].value == []
        assert len(at.number_input) == 0
        assert _button(at, "Analyze Portfolio").disabled is True

        # session_state itself reflects the reset, not just the widgets.
        # AppTest's session_state wrapper (SafeSessionState) doesn't support
        # .keys() directly — .filtered_state gives a plain dict view.
        state = at.session_state.filtered_state
        assert state["portfolio_version"] == version_before + 1
        assert not any(
            key.startswith(f"weight_{version_before}_") for key in state
        )
        assert state.get("run_in_progress") in (None, False)
