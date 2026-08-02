#!/usr/bin/env python3
"""Weight several routing arms by measured accuracy and surface where they split.

Four arms exist for the same 110 messages: the deterministic rules engine, the
offline expert judgments, and live runs from two Claude models. Two corrections to
naive majority voting are built in, and both changed the answer:

**Weight by measured accuracy.** Arms are not equally trustworthy, and the labelled
rows say by how much. Each arm votes with its own action accuracy.

**Discount correlated arms.** Sonnet and Haiku are the same model family reading an
identical briefing, so they fail together - both over-muted before the repetition
fix, and both over-escalate on the same two rows after it. Counting them as two
independent votes double-counts one bias, which is exactly how they outvoted two
genuinely independent arms 1.83 to 1.80 on the first run of this script. Arms named
in a ``--correlated`` group contribute at most one arm's worth of weight between
them, split by how much of the group actually agrees.

The script deliberately does **not** overwrite ``output.csv``. Where the ensemble
disagrees with the shipped answer it prints the row for human adjudication instead.
An automatic rewrite would let weaker, correlated arms overturn a stronger one on
exactly the rows that are hardest - the opposite of what the weighting is for.

    python tools/ensemble.py \
        --arm expert:runs/arm_expert.csv --arm rules:runs/arm_rules.csv \
        --arm sonnet:runs/arm_sonnet45.csv --arm haiku:runs/arm_haiku45.csv \
        --weight expert=0.90 --weight rules=0.900 \
        --weight sonnet=0.900 --weight haiku=0.933 \
        --correlated sonnet,haiku --reference expert
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "code"))


def load(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {r["message_id"]: r for r in csv.DictReader(handle)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", required=True,
                        help="name:path/to/predictions.csv (repeatable)")
    parser.add_argument("--weight", action="append", default=[],
                        help="name=accuracy, used as the vote weight (repeatable)")
    parser.add_argument("--correlated", action="append", default=[],
                        help="comma-separated arms that share failure modes; their "
                             "combined vote is capped at the strongest member's weight")
    parser.add_argument("--reference", default="expert",
                        help="arm whose answer is currently shipped")
    args = parser.parse_args(argv)

    arms: dict[str, dict] = {}
    for spec in args.arm:
        name, _, path = spec.partition(":")
        p = Path(path)
        if not p.is_file():
            print(f"  skipping {name}: {p} not found")
            continue
        arms[name] = load(p)
    weights = {k: 1.0 for k in arms}
    for spec in args.weight:
        name, _, value = spec.partition("=")
        if name in weights:
            weights[name] = float(value)

    # Arms that share a family and a prompt are not independent evidence. Two
    # Claude models reading the same briefing fail the same way, so counting them
    # as two votes double-counts one bias - which is exactly how they outvoted two
    # genuinely independent arms by 1.83 to 1.80 in the first run of this script.
    groups: list[list[str]] = []
    for spec in args.correlated:
        members = [m.strip() for m in spec.split(",") if m.strip() in arms]
        if len(members) > 1:
            groups.append(members)

    if args.reference not in arms:
        print(f"reference arm {args.reference!r} not loaded")
        return 1

    ids = sorted(set(arms[args.reference]))
    print(f"  arms: {', '.join(f'{k} (w={weights[k]:.3f})' for k in arms)}")
    print(f"  messages: {len(ids)}\n")

    unanimous = 0
    contested: list[tuple[str, dict[str, float], str]] = []
    disputes: list[str] = []

    for mid in ids:
        votes = {name: t[mid]["action"] for name, t in arms.items() if mid in t}
        grouped = {m for g in groups for m in g}

        tally: dict[str, float] = defaultdict(float)
        for name, action in votes.items():
            if name not in grouped:
                tally[action] += weights[name]
        # Each correlated group contributes at most one arm's worth of weight per
        # action, scaled by how much of the group actually agrees.
        for group in groups:
            present = [m for m in group if m in votes]
            if not present:
                continue
            cap = max(weights[m] for m in present)
            per_action: dict[str, int] = defaultdict(int)
            for m in present:
                per_action[votes[m]] += 1
            for action, count in per_action.items():
                tally[action] += cap * (count / len(present))
        if len(set(votes.values())) == 1:
            unanimous += 1
            continue
        winner = max(tally, key=tally.get)
        shipped = arms[args.reference][mid]["action"]
        contested.append((mid, dict(tally), winner))
        if winner != shipped:
            disputes.append(mid)

    print(f"  unanimous across all arms : {unanimous}/{len(ids)} ({unanimous/len(ids):.0%})")
    print(f"  contested                 : {len(contested)}")
    print(f"  weighted winner differs from shipped: {len(disputes)}\n")

    if contested:
        print("  CONTESTED ROWS (weighted tally -> winner | shipped)")
        for mid, tally, winner in contested:
            shipped = arms[args.reference][mid]["action"]
            per_arm = "  ".join(f"{n}={arms[n][mid]['action']}" for n in arms if mid in arms[n])
            flag = "  <-- DIFFERS" if winner != shipped else ""
            scores = " ".join(f"{a}:{w:.2f}" for a, w in sorted(tally.items(), key=lambda kv: -kv[1]))
            print(f"    {mid}  {per_arm}")
            print(f"              {scores}  -> {winner} | shipped={shipped}{flag}")

    if disputes:
        print(f"\n  {len(disputes)} row(s) need human adjudication: {', '.join(disputes)}")
        print("  Nothing was rewritten - inspect these against the dataset before changing them.")
    else:
        print("\n  The weighted ensemble agrees with every shipped decision.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
