"""
llm.py — LLM interaction via GPT-4o-mini.
Builds the tutor prompt and handles conversation history.
"""

import json

from services.processor import get_openai_client


def _parse_gpt_json(raw: str) -> dict:
    """
    Parse a JSON string that GPT may have wrapped in markdown code fences.
    Strips ```json ... ``` or ``` ... ``` before parsing.
    Returns a safe fallback dict if parsing still fails.
    """
    text = raw.strip()
    # Remove opening fence: ```json or ```
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]  # drop the first line (``` or ```json)
    # Remove closing fence
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "concepts_assessed": [],
            "concepts_not_covered": [],
            "summary": "Session complete. Report could not be generated.",
        }

# ---------------------------------------------------------------------------
# System prompt — context-locked tutor persona
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Paideia, a strict academic tutor. Your knowledge is strictly \
limited to the retrieved context below from the student's own uploaded notes. \
If a question cannot be answered from this context, say: \
"I don't see that in your notes yet, but based on what you have uploaded, here is what I can tell you." \
Never teach from general knowledge. Only from the context provided. \
Speak in short natural bursts of 2 to 3 sentences. Always end with a question. \
Never use lists or formatting."""


def build_tutor_prompt(context: str, student_message: str) -> list[dict]:
    """
    Build the messages list for the chat completion call.

    The system message injects the retrieved context so GPT-4o-mini
    is grounded entirely in the student's own notes.
    history is NOT passed here — the caller appends it between the
    system message and the final user message so the model sees the
    full conversation thread.
    """
    system_content = SYSTEM_PROMPT

    if context:
        system_content += f"\n\n--- Retrieved context from student notes ---\n{context}"
    else:
        system_content += (
            "\n\n--- No relevant context was found in the student's notes. ---"
            "\nTell the student you could not find relevant material and encourage "
            "them to upload their notes first."
        )

    return [{"role": "system", "content": system_content}]


async def get_llm_response(
    student_message: str,
    context: str,
    history: list[dict],
) -> str:
    """
    Call GPT-4o-mini and return the assistant's reply as a plain string.

    history: list of {"role": "user"|"assistant", "content": "..."} dicts
             representing the conversation so far (not including the current message).
    context: the RAG-retrieved text from ChromaDB.
    """
    client = get_openai_client()

    # Build messages: system prompt (with context) + prior history + new user turn
    messages = build_tutor_prompt(context, student_message)
    messages.extend(history)
    messages.append({"role": "user", "content": student_message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.4,      # low temperature keeps answers factual and grounded
        max_tokens=300,       # short replies suit voice — 2-3 sentences
    )

    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Session report — mastery assessment after conversation ends
# ---------------------------------------------------------------------------

# REPORT_PROMPT is built inside generate_session_report as an f-string
# so that {{ }} can escape the literal JSON braces in the example
# while {transcript}, {concepts_touched_str}, {full_map_str} remain live variables.


async def generate_session_report(
    conversation_history: list[dict],
    concepts_touched: list[dict],
    full_concept_map: list[dict],
) -> dict:
    """
    Assess student mastery after a session ends.

    conversation_history: list of {role, content} dicts from the session
    concepts_touched: list of {lesson, concept, chunk_content} dicts —
                      every concept retrieved from the notes during the session
    full_concept_map: list of {lesson, concept} dicts — everything in the
                      student's uploaded notes (from get_concept_summary)

    Returns a report dict with concepts_assessed, concepts_not_covered, summary.
    Returns a minimal report if the session was too short to assess.
    """
    # Graceful handling for empty sessions
    if not conversation_history:
        return {
            "concepts_assessed": [],
            "concepts_not_covered": [c.get("concept", "") for c in full_concept_map],
            "summary": "The session ended before any conversation took place.",
        }

    # Build a readable transcript from the history
    transcript_lines = []
    for turn in conversation_history:
        role = "Student" if turn.get("role") == "user" else "Paideia"
        transcript_lines.append(f"{role}: {turn.get('content', '')}")
    transcript = "\n".join(transcript_lines)

    # Deduplicate concepts_touched by lesson+concept pair for the prompt
    seen = set()
    unique_concepts = []
    for c in concepts_touched:
        key = (c.get("lesson", ""), c.get("concept", ""))
        if key not in seen:
            seen.add(key)
            unique_concepts.append({"lesson": key[0], "concept": key[1]})

    # Format the concept lists as readable strings for the prompt
    concepts_touched_str = "\n".join(
        f"- {c['concept']} (from: {c['lesson']})" for c in unique_concepts
    ) or "None"

    full_map_str = "\n".join(
        f"- {c.get('concept', '')} (from: {c.get('lesson', '')})" for c in full_concept_map
    ) or "No notes uploaded"

    # Build the prompt as an f-string so {{ }} escapes literal JSON braces
    # while {transcript}, {concepts_touched_str}, {full_map_str} are substituted normally.
    prompt = f"""You are Paideia, an AI academic tutor. A tutoring session just ended.
Below is the full conversation transcript and the list of concepts that came up during the session.
Assess the student's understanding of each concept based solely on how they answered.

Return ONLY valid JSON in exactly this format — no markdown, no explanation:
{{
  "concepts_assessed": [
    {{
      "concept": "concept name",
      "lesson": "lesson name",
      "mastery": "strong or developing or weak",
      "evidence": "one sentence explaining why, based on the student's actual responses"
    }}
  ],
  "concepts_not_covered": ["concept name", "concept name"],
  "summary": "2-3 sentence summary written as Paideia speaking directly to the student"
}}

Mastery definitions:
- strong: student answered correctly and showed clear understanding
- developing: student had partial understanding or needed hints
- weak: student struggled, gave wrong answers, or did not engage with the concept

CONVERSATION TRANSCRIPT:
{transcript}

CONCEPTS TOUCHED IN THIS SESSION:
{concepts_touched_str}

ALL CONCEPTS IN STUDENT'S NOTES (for concepts_not_covered):
{full_map_str}
"""

    client = get_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1500,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    return _parse_gpt_json(raw)
