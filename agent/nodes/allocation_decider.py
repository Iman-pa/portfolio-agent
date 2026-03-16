import json
from pathlib import Path

import yaml
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from agent.state import PortfolioState

# Resolve the path to the YAML prompt file relative to this file's location.
# __file__ is allocation_decider.py inside agent/nodes/.
# Two .parent calls walk up to the project root, then we descend into prompts/.
_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "allocation_decider.yaml"

# Instantiate the Gemini model once at import time so it is reused across
# every call to allocation_decider() rather than being recreated each run.
# temperature=0 makes the output deterministic — critical for structured JSON.
_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)


def _load_prompt(
    research_block: str,
    strategy_instruction: str,
    macro_context: dict[str, float],
) -> list:
    """Read the YAML file, fill every placeholder, and return a message list.

    Three placeholders exist in the user prompt:
      {research_results}      — the formatted per-ticker research block
      {strategy_instruction}  — the strategy-specific paragraph from the YAML
      {vix}, {yield_10y}, {spy_price} — the three macro indicator values
    """
    # Open and parse the YAML file into a plain Python dict.
    # yaml.safe_load() is used rather than yaml.load() to prevent execution
    # of arbitrary Python objects embedded in the file.
    with open(_PROMPT_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Fill all placeholders in the user prompt string in a single .format() call.
    # Python's str.format() replaces every {key} with the matching keyword argument.
    # The macro values are unpacked from the dict using ** so we don't have to
    # list each key individually — any key in macro_context becomes an available
    # placeholder name (e.g. {vix}, {yield_10y}, {spy_price}).
    user_text = raw["user"].format(
        research_results=research_block,
        strategy_instruction=strategy_instruction,
        **macro_context,
    )

    # Wrap each string in the appropriate LangChain message type.
    # SystemMessage sets the model's persona/behaviour for the entire conversation.
    # HumanMessage is the user turn that contains the actual decision task.
    return [
        SystemMessage(content=raw["system"]),
        HumanMessage(content=user_text),
    ]


def _build_research_block(research_results: dict[str, str]) -> str:
    """Format the research_results dict into a readable labelled string.

    Produces one section per ticker separated by blank lines, e.g.:
        --- AAPL ---
        Research report for AAPL: ...

        --- NVDA ---
        Research report for NVDA: ...
    """
    sections = []
    for ticker, report in research_results.items():
        sections.append(f"--- {ticker} ---\n{report}")

    # "\n\n".join() places a blank line between each ticker section so the
    # model can clearly distinguish where one report ends and the next begins.
    return "\n\n".join(sections)


def _get_strategy_instruction(raw: dict, strategy: str) -> str:
    """Look up the strategy-specific instruction block from the parsed YAML.

    The YAML stores each strategy under a key like "strategy_conservative".
    We build that key by prepending "strategy_" to the user-supplied strategy
    string, then look it up in the parsed dict.

    Raises KeyError with a clear message if the strategy name is not found,
    which surfaces a helpful error in the Streamlit UI rather than a cryptic
    Python traceback.
    """
    # Build the YAML key by convention: "strategy_" + the strategy name.
    # e.g. "balanced" → "strategy_balanced"
    yaml_key = f"strategy_{strategy}"

    if yaml_key not in raw:
        # Listing the valid keys in the error message helps the developer
        # immediately identify whether a typo or a missing YAML section is
        # the cause of the problem.
        valid = [k for k in raw if k.startswith("strategy_")]
        raise KeyError(
            f"Unknown strategy '{strategy}'. "
            f"Valid options: {[k.replace('strategy_', '') for k in valid]}"
        )

    return raw[yaml_key]


def _parse_allocations(raw_text: str) -> dict[str, dict]:
    """Parse the model's richer JSON response string into a Python dict.

    The expected structure is:
        {
            "AAPL": {"allocation": 35.0, "confidence": 82, "reason": "..."},
            "NVDA": {"allocation": 40.0, "confidence": 90, "reason": "..."},
        }

    .strip() removes any leading/trailing whitespace the model may have added.
    json.loads() raises JSONDecodeError on malformed output, which propagates
    to the Streamlit error handler rather than silently returning bad data.
    The parsed dict is returned as-is — each value is already a dict with the
    three required keys, so no further transformation is needed here.
    """
    return json.loads(raw_text.strip())


def allocation_decider(state: PortfolioState) -> dict:
    """Call Gemini to decide portfolio allocation percentages.

    Reads research_results, strategy, and macro_context from state, assembles
    the prompt with all three injected, calls Gemini, and parses the JSON
    response into the allocations dict.
    """
    # Read the YAML once so we can pass `raw` to both _get_strategy_instruction
    # and _load_prompt without opening the file twice.
    with open(_PROMPT_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Format all research reports into a single labelled string.
    research_block = _build_research_block(state["research_results"])

    # Look up the strategy instruction paragraph that matches the user's choice.
    # state["strategy"] is a string like "conservative" or "aggressive" —
    # set by the Streamlit dropdown and passed in via graph.invoke().
    strategy_instruction = _get_strategy_instruction(raw, state["strategy"])

    # Build the full message list with all three placeholders filled.
    # state["macro_context"] is the dict written by macro_fetcher, e.g.
    # {"vix": 18.3, "yield_10y": 4.6, "spy_price": 542.1}.
    messages = _load_prompt(
        research_block=research_block,
        strategy_instruction=strategy_instruction,
        macro_context=state["macro_context"],
    )

    # Send the assembled messages to Gemini and wait for the response.
    # .invoke() is synchronous — it blocks until the model replies.
    response = _llm.invoke(messages)

    # response.content is the raw string from the model, expected to be a
    # JSON object like '{"AAPL": 40.0, "NVDA": 35.0, "TSLA": 25.0}'.
    allocations = _parse_allocations(response.content)

    # Return only the field this node writes; LangGraph merges it into state.
    return {"allocations": allocations}
