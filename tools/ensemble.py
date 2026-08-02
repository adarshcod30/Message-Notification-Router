#!/usr/bin/env python3
"""Weight several routing arms by measured accuracy and surface where they split.

Four independent arms now exist for the same 110 messages: the deterministic rules
engine, the offline expert judgments, and live runs from two Claude models. Simple
majority voting would treat them as equally trustworthy, which the measurements say
they are not - so each arm's vote is weighted by its own action accuracy on the 30
labelled rows.

The script deliberately does **not** overwrite ``output.csv``. Where the weighted
ensemble disagrees with the shipped answer, that row is printed for human
adjudication instead. An automatic rewrite would let a majority of weaker arms
outvote a stronger one on exactly the rows that are hardest, which is the opposite
of what the weighting is for.

    python tools/ensemble.py --arm expert:output.csv \
                             --arm sonnet:runs/second_opinion_sonnet45.csv \
                             --arm haiku:/tmp/out_haiku.csv \
                             --weight expert=0.90 --weight sonnet=0.867 --weight haiku=0.80
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
        tally: dict[str, float] = defaultdict(float)
        votes = {}
        for name, table in arms.items():
            row = table.get(mid)
            if row is None:
                continue
            votes[name] = row["action"]
            tally[row["action"]] += weights[name]
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
