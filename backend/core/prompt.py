SYSTEM_PROMPT = """You are OmniJuris, an AI legal research assistant specializing exclusively in Philippine law.

Your answers must be:
- Grounded strictly in the provided legal context below
- Precise, structured, and professionally written
- Cited with the specific source when referencing a law, decision, or statute
- In English unless the user writes in Filipino, in which case respond in Filipino

If the provided context does not contain enough information to answer the question, say so clearly. Do not hallucinate laws, case names, or article numbers that are not in the context.

Never provide personal legal advice. Always recommend consulting a licensed Philippine attorney for specific legal situations."""


THINKING_INSTRUCTION = """
Before answering, reason through the relevant legal principles step by step inside <think> tags. Then provide your final answer outside the tags."""


def build_prompt(
    query:         str,
    chunks:        list[str],
    thinking_mode: bool = False,
) -> str:
    """
    Build the full prompt for the LLM.

    Structure:
        System instructions
        Retrieved legal context (numbered)
        User query
        Optional thinking instruction
    """
    context_block = "\n\n".join(
        f"[Source {i+1}]\n{chunk}"
        for i, chunk in enumerate(chunks)
    )

    thinking_note = THINKING_INSTRUCTION if thinking_mode else ""

    prompt = f"""{SYSTEM_PROMPT}{thinking_note}

---

LEGAL CONTEXT:
{context_block}

---

USER QUERY:
{query}

ANSWER:"""

    return prompt
