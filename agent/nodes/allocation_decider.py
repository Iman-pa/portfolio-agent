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
# `model` selects the specific Gemini variant.
# `temperature=0` makes the output deterministic — important for structured
# JSON responses where we don't want creative variation.
_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)


def _load_prompt(research_block: str) -> list:
    """Read the YAML file, fill in the placeholder, and return a message list.

    Returns a list of LangChain message objects ready to be passed to the LLM.
    """
    # Open and parse the YAML file into a plain Python dict with keys
    # "system" and "user", each containing a multi-line string.
    with open(_PROMPT_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Fill the {research_results} placeholder in the user prompt with the
    # actual research block string.  .format() replaces every occurrence of
    # {research_results} with the value we pass.
    user_text = raw["user"].format(research_results=research_block)

    # Wrap each string in the appropriate LangChain message type.
    # SystemMessage sets the behaviour/persona of the model.
    # HumanMessage represents the user turn that contains the actual task.
    return [
        SystemMessage(content=raw["system"]),
        HumanMessage(content=user_text),
    ]


def _build_research_block(research_results: dict[str, str]) -> str:
    """Format the research_results dict into a readable string for the prompt.

    Produces one labelled section per ticker, e.g.:
        --- AAPL ---
        Research report for AAPL: ...

        --- NVDA ---
        Research report for NVDA: ...
    """
    sections = []
    for ticker, report in research_results.items():
        # Each section is a header line followed by the report text.
        sections.append(f"--- {ticker} ---\n{report}")

    # Join all sections with a blank line between them for readability.
    return "\n\n".join(sections)


def _parse_allocations(raw_text: str) -> dict[str, float]:
    """Extract the JSON allocation dict from the model's raw response text.

    The prompt instructs Gemini to return only a JSON object, but we strip
    whitespace defensively in case the model adds a newline or space.
    """
    # json.loads() parses a JSON string into a Python dict.
    # If the model returns malformed JSON this will raise json.JSONDecodeError,
    # which is intentionally left to propagate so the caller sees a clear error.
    return json.loads(raw_text.strip())


def allocation_decider(state: PortfolioState) -> dict:
    """Call Gemini to decide portfolio allocation percentages.

    Reads the fully populated `research_results` from state, formats them
    into the prompt template, calls the Gemini LLM, and parses the returned
    JSON into the `allocations` dict.
    """
    # Format all research reports into a single readable string that will be
    # injected into the {research_results} placeholder in the prompt.
    research_block = _build_research_block(state["research_results"])

    # Load the YAML prompt, fill in the placeholder, and get message objects.
    messages = _load_prompt(research_block)

    # Send the messages to Gemini.  .invoke() is a synchronous call that
    # blocks until the model responds and returns an AIMessage object.
    response = _llm.invoke(messages)

    # response.content is the raw string the model returned, e.g.:
    # '{"AAPL": 40.0, "NVDA": 35.0, "TSLA": 25.0}'
    allocations = _parse_allocations(response.content)

    # Return only the field this node writes.  LangGraph merges it into state.
    return {"allocations": allocations}
