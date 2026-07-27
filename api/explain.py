"""Model C — grounded explanation of a Model A prediction (Phase 4).

Hard separation (STRATEGY.md §3): Model C must never create or alter the number.
The calibrated probability and uncertainty are computed by Model A (api.service)
and copied verbatim into the response; the LLM, if any, only phrases the context
retrieved from Apollo's docs. Even when an LLM is used, the exact probability line
and the disclaimer are appended by code, so narrative cannot inflate certainty.
"""

from __future__ import annotations

from api import service
from api.knowledge import Retriever
from engine.codes import describe_scenario

SYSTEM = (
    "You explain a fixed, pre-computed risk estimate to an analyst. You must NOT "
    "state, change, or imply any probability other than the one given. Do not invent "
    "facts; use only the provided context. Be concise, cautious, and non-operational. "
    "This is retrospective, probabilistic decision-support, never a forecast that an "
    "attack will happen and never a basis for action against any person or group."
)


def _prob_line(pred: dict) -> str:
    return (
        f"Estimated probability of >=1 fatality: {pred['probability']:.2f} "
        f"(uncertainty {pred['uncertainty_low']:.2f}-{pred['uncertainty_high']:.2f})."
    )


def _template_explanation(pred: dict, scenario: str, evidence: list) -> str:
    """Deterministic, fully-grounded explanation used by default and as fallback."""
    lines = [scenario, _prob_line(pred)]
    if evidence:
        lines.append("Context from Apollo's documentation:")
        for ev in evidence:
            snippet = ev.snippet[:240].rstrip()
            lines.append(f"- ({ev.source}) {snippet}")
    lines.append(pred["disclaimer"])
    return "\n".join(lines)


def _llm_user_prompt(pred: dict, scenario: str, evidence: list) -> str:
    ctx = "\n".join(f"- ({ev.source}) {ev.snippet}" for ev in evidence) or "(none)"
    return (
        f"Scenario: {scenario}\n"
        f"{_prob_line(pred)}\n\n"
        f"Context (quote/paraphrase only from here):\n{ctx}\n\n"
        "Write 2-4 sentences explaining what drives an estimate like this and how to "
        "read it responsibly. Do not state any different probability."
    )


def explain(bundle: dict, features: dict, retriever: Retriever | None, llm=None) -> dict:
    # 1. The number comes from Model A — authoritative, never from the LLM.
    pred = service.predict(bundle, features)

    # 2. Ground: retrieve both "what drives this scenario" and, always, the
    #    responsible-use / calibration context — merged and de-duplicated.
    scenario = describe_scenario(features)
    evidence = []
    if retriever is not None:
        drivers = retriever.retrieve(f"{scenario} lethality weapon attack type", k=2)
        caveats = retriever.retrieve(
            "calibration uncertainty limitations fairness how to read a prediction", k=2
        )
        seen = set()
        for ev in [*drivers, *caveats]:
            if ev.source not in seen:
                seen.add(ev.source)
                evidence.append(ev)

    # 3. Phrase: template by default; LLM only if configured and it succeeds.
    generated_by = "template"
    narrative = _template_explanation(pred, scenario, evidence)
    if llm is not None:
        text = llm.generate(SYSTEM, _llm_user_prompt(pred, scenario, evidence))
        if text:
            # Re-anchor the exact number + disclaimer so narrative can't override them.
            narrative = f"{text}\n\n{_prob_line(pred)}\n{pred['disclaimer']}"
            generated_by = getattr(llm, "name", "llm")

    # 4. Numeric fields are copied straight from Model A's output.
    return {
        "probability": pred["probability"],
        "uncertainty_low": pred["uncertainty_low"],
        "uncertainty_high": pred["uncertainty_high"],
        "target": pred["target"],
        "disclaimer": pred["disclaimer"],
        "scenario": scenario,
        "explanation": narrative,
        "evidence": [{"source": ev.source, "snippet": ev.snippet, "score": ev.score}
                     for ev in evidence],
        "generated_by": generated_by,
    }
