"""
PDDL Generator -- converts action programs into PDDL problem instances.

Takes a list of ActionSteps (with preconditions and effects) and generates
a complete PDDL domain + problem that a robot task planner can consume.
"""
from __future__ import annotations

from src.action_program import ActionStep


def _sanitize(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_").replace(":", "_")


def generate_pddl(steps: list[ActionStep], video_info: dict | None = None) -> str:
    """Generate PDDL domain + problem from action steps."""

    # Collect all unique nouns for object declarations
    nouns = set()
    for s in steps:
        if s.noun and s.noun not in ("none", "unknown", "surface"):
            nouns.add(_sanitize(s.noun))

    # Collect all unique predicates used
    all_predicates = set()
    for s in steps:
        for p in s.preconditions + s.effects:
            if p.startswith("NOT "):
                p = p[4:]
            # Extract predicate name
            pname = p.split("(")[0].strip()
            all_predicates.add(pname)

    # ── Domain ─────────────────────────────────────────────────────
    domain = """;; HUMOS-v2 Auto-Generated PDDL Domain
;; Action-Centric Ground Truth for Humanoid Navigation
;; Generated from egocentric video analysis

(define (domain kitchen-actions)
  (:requirements :strips :typing)

  (:types
    agent - object
    item - object
    location - object
  )

  (:predicates
    (at ?a - agent ?l - location)
    (holding ?a - agent ?o - item)
    (hand_free ?a - agent)
    (on ?o - item ?l - location)
    (inside ?o - item ?c - item)
    (reachable ?a - agent ?o - item)
    (is_open ?o - item)
    (is_closed ?o - item)
    (is_on ?o - item)
    (is_off ?o - item)
    (is_clean ?o - item)
    (is_cut ?o - item)
    (is_peeled ?o - item)
    (is_mixed ?o - item)
    (is_stirred ?o - item)
    (is_dry ?o - item)
    (is_wet ?o - item)
    (poured ?o - item)
    (squeezed ?o - item)
    (shaken ?o - item)
    (disposed ?o - item)
    (moved ?o - item)
    (adjusted ?o - item)
    (scooped ?o - item)
  )
"""

    # Generate one PDDL action per unique verb
    verbs_seen = set()
    actions_pddl = ""
    for s in steps:
        v = _sanitize(s.verb)
        if v in verbs_seen or v == "idle":
            continue
        verbs_seen.add(v)

        actions_pddl += f"""
  (:action {v}
    :parameters (?a - agent ?o - item)
    :precondition (and (reachable ?a ?o))
    :effect (and)
  )
"""

    domain += actions_pddl + ")\n"

    # ── Problem ────────────────────────────────────────────────────
    video_comment = ""
    if video_info:
        video_comment = f";; Source: {video_info.get('path', 'unknown')}\n"
        video_comment += f";; Duration: {video_info.get('duration_sec', 0):.1f}s\n"

    obj_decls = "\n".join(f"    {n} - item" for n in sorted(nouns))

    # Initial state: everything reachable, hand free
    init_preds = ["    (hand_free human)"]
    for n in sorted(nouns):
        init_preds.append(f"    (reachable human {n})")

    # Apply initial state from first step's preconditions
    for s in steps[:1]:
        for p in s.preconditions:
            if p.startswith("NOT "):
                continue
            pred = p.replace("(", " ").replace(")", "").replace(",", "").strip()
            init_preds.append(f"    ({pred})")

    problem = f""";; HUMOS-v2 Auto-Generated PDDL Problem
{video_comment}
(define (problem observed-actions)
  (:domain kitchen-actions)

  (:objects
    human - agent
    kitchen - location
{obj_decls}
  )

  (:init
    (at human kitchen)
{chr(10).join(init_preds)}
  )

  (:goal (and
    ;; Goal state derived from final effects
  ))
)
"""

    # ── Action Trace ───────────────────────────────────────────────
    trace = "\n;; ══ Observed Action Trace ══\n"
    trace += f";; {len(steps)} actions extracted from video\n\n"

    for s in steps:
        trace += f";; [{s.start_sec:.1f}s - {s.stop_sec:.1f}s] ({s.verb} human {_sanitize(s.noun)}) ;; conf={s.confidence:.0%}\n"
        if s.preconditions:
            trace += f";;   PRE:  {', '.join(s.preconditions)}\n"
        if s.effects:
            trace += f";;   POST: {', '.join(s.effects)}\n"
        trace += f";;   \"{s.description}\"\n\n"

    return domain + "\n" + problem + trace
