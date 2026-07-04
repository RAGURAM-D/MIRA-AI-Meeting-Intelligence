"""
Multi-Agent Summarization Module - Chapter 4.3
LangChain + LangGraph pipeline with 4 specialized agents:
  Agent 1 - Transcript Cleaner
  Agent 2 - Meeting Summarizer
  Agent 3 - Action Item Extractor
  Agent 4 - Decision Extractor
Orchestrated as a directed acyclic graph via LangGraph.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langgraph.prebuilt import ToolNode, tools_condition

# LLM: GPT-4o-mini via OpenRouter (Chapter 2.2)
def get_llm():
    return ChatOpenAI(
        model="openai/gpt-4o-mini",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.2,
    )


def format_transcript(segments: list[dict]) -> str:
    """Convert attributed segments list to readable transcript string."""
    lines = []
    current_speaker = None
    current_text = []

    for seg in segments:
        if seg["speaker"] != current_speaker:
            if current_speaker and current_text:
                lines.append(f"{current_speaker}: {' '.join(current_text)}")
            current_speaker = seg["speaker"]
            current_text = [seg["text"]]
        else:
            current_text.append(seg["text"])

    if current_speaker and current_text:
        lines.append(f"{current_speaker}: {' '.join(current_text)}")

    return "\n\n".join(lines)


# ─── Agent 1: Transcript Cleaner ───────────────────────────────────────────

def agent_clean_transcript(raw_transcript: str) -> str:
    """
    Agent 1 - Transcript Cleaner
    Removes filler words, corrects ASR errors, structures speaker turns.
    """
    print("[Agent 1] Cleaning transcript...")
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a professional transcript editor. Clean the following meeting transcript by:\n"
            "1. Removing filler words (um, uh, like, you know)\n"
            "2. Correcting obvious speech recognition errors\n"
            "3. Preserving all speaker labels exactly (Speaker 1, Speaker 2, etc.)\n"
            "4. Keeping all content — do not summarize or remove information\n"
            "Return only the cleaned transcript, nothing else."
        )),
        ("human", "{transcript}"),
    ])

    chain = prompt | llm
    result = chain.invoke({"transcript": raw_transcript})
    return result.content


# ─── Agent 2: Meeting Summarizer ───────────────────────────────────────────

def agent_summarize(cleaned_transcript: str) -> str:
    """
    Agent 2 - Meeting Summarizer
    Generates 150-250 word executive summary.
    """
    print("[Agent 2] Generating executive summary...")
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert meeting summarizer. Based on the transcript, write a concise executive summary "
            "of 150 to 250 words covering:\n"
            "- Main topics discussed\n"
            "- Overall conclusions\n"
            "- Who participated\n"
            "Use only information explicitly present in the transcript. "
            "Return only the summary paragraph, nothing else."
        )),
        ("human", "{transcript}"),
    ])

    chain = prompt | llm
    result = chain.invoke({"transcript": cleaned_transcript})
    return result.content


# ─── Agent 3: Action Item Extractor ────────────────────────────────────────

def agent_extract_action_items(cleaned_transcript: str) -> list[dict]:
    """
    Agent 3 - Action Item Extractor
    Identifies tasks, assignees, and deadlines.
    """
    print("[Agent 3] Extracting action items...")
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an action item extraction specialist. Identify all tasks, commitments, and follow-ups "
            "from this meeting transcript.\n\n"
            "For each action item return a line in this exact format:\n"
            "TASK: <task description> | OWNER: <speaker name> | DEADLINE: <deadline or 'Not specified'>\n\n"
            "Extract only explicit commitments. Return each action item on a new line."
        )),
        ("human", "{transcript}"),
    ])

    chain = prompt | llm
    result = chain.invoke({"transcript": cleaned_transcript})

    action_items = []
    for line in result.content.strip().split("\n"):
        line = line.strip()
        if "TASK:" in line and "OWNER:" in line:
            try:
                parts = line.split("|")
                task = parts[0].replace("TASK:", "").strip()
                owner = parts[1].replace("OWNER:", "").strip()
                deadline = parts[2].replace("DEADLINE:", "").strip() if len(parts) > 2 else "Not specified"
                action_items.append({"task": task, "owner": owner, "deadline": deadline})
            except Exception:
                continue

    return action_items


# ─── Agent 4: Decision Extractor ───────────────────────────────────────────

def agent_extract_decisions(cleaned_transcript: str) -> list[str]:
    """
    Agent 4 - Decision Extractor
    Identifies formal decisions and agreements made during the meeting.
    """
    print("[Agent 4] Extracting key decisions...")
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a decision identification specialist. Extract all formal decisions, agreements, "
            "and resolved discussion points from this meeting transcript.\n\n"
            "Return each decision on a new line starting with 'DECISION: '.\n"
            "Only include clear, explicit decisions — not ongoing discussions."
        )),
        ("human", "{transcript}"),
    ])

    chain = prompt | llm
    result = chain.invoke({"transcript": cleaned_transcript})

    decisions = []
    for line in result.content.strip().split("\n"):
        line = line.strip()
        if line.startswith("DECISION:"):
            decisions.append(line.replace("DECISION:", "").strip())

    return decisions


# ─── LangGraph Orchestration ────────────────────────────────────────────────

class PipelineState(TypedDict):
    raw_transcript: str
    cleaned_transcript: str
    summary: str
    action_items: list
    decisions: list


def build_langgraph_pipeline():
    """
    Build LangGraph DAG:
    Agent1 (clean) → parallel: Agent2, Agent3, Agent4 → Compiler
    """
    graph = StateGraph(PipelineState)

    def node_clean(state: PipelineState) -> PipelineState:
        cleaned = agent_clean_transcript(state["raw_transcript"])
        return {**state, "cleaned_transcript": cleaned}

    def node_parallel_agents(state: PipelineState) -> PipelineState:
        """Run Agents 2, 3, 4 in parallel using ThreadPoolExecutor."""
        transcript = state["cleaned_transcript"]
        results = {}

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(agent_summarize, transcript): "summary",
                executor.submit(agent_extract_action_items, transcript): "action_items",
                executor.submit(agent_extract_decisions, transcript): "decisions",
            }
            for future in as_completed(futures):
                key = futures[future]
                results[key] = future.result()

        return {**state, **results}

    graph.add_node("clean", node_clean)
    graph.add_node("extract", node_parallel_agents)
    graph.add_edge("clean", "extract")
    graph.add_edge("extract", END)
    graph.set_entry_point("clean")

    return graph.compile()


def run_agent_pipeline(attributed_segments: list[dict]) -> dict:
    """
    Main entry point for the multi-agent pipeline.
    Accepts attributed segments, returns compiled report dict.
    """
    raw_transcript = format_transcript(attributed_segments)
    print(f"[Pipeline] Starting multi-agent pipeline. Transcript length: {len(raw_transcript)} chars")

    pipeline = build_langgraph_pipeline()
    final_state = pipeline.invoke({
        "raw_transcript": raw_transcript,
        "cleaned_transcript": "",
        "summary": "",
        "action_items": [],
        "decisions": [],
    })

    # Extract unique attendees from segments
    attendees = sorted(set(seg["speaker"] for seg in attributed_segments))

    report_data = {
        "summary": final_state["summary"],
        "action_items": final_state["action_items"],
        "decisions": final_state["decisions"],
        "attendees": attendees,
        "transcript": final_state["cleaned_transcript"],
    }

    print(f"[Pipeline] Complete. "
          f"{len(report_data['action_items'])} action items, "
          f"{len(report_data['decisions'])} decisions.")

    return report_data
