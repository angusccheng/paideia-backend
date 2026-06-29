"""
llm.py — LLM interaction via GPT-4o-mini.
Builds the tutor prompt and handles conversation history.
"""

from services.processor import get_openai_client

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
