#!/usr/bin/env python3
"""Validate predictions against ground truth transferred from the labelled sample.

Ten rows in ``messages.csv`` duplicate a row in ``sample_messages.csv`` closely and
for the *same recipient* - the same routing decision under a different id. Those
carry real gold labels on the actual test file, which is otherwise unlabelled.

Recipient identity is part of the match on purpose. The same text to a different
user is a genuinely different decision in this dataset (``msg_103`` vs ``msg_104``
are byte-identical and correctly routed opposite ways), so matching on text alone
would manufacture false gold.

**Read the result with its caveat.** Arms that were authored or tuned with the
labelled rows in view - the expert judgments and the deterministic rules - are not
blind here. The live model arms never see gold labels, so their score is the only
fully blind one. Where an arm disagrees with gold, though, gold decides regardless
of who saw what.

    python tools/gold_transfer.py --arm expert:runs/arm_expert.csv --blind sonnet45,haiku45
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "code"))

from router.context_store import ContextStore  # noqa: E402
from router.signals import text_similarity  # noqa: E402

THRESHOLD = 0.85


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", required=True, help="name:path.csv")
    parser.add_argument("--blind", default="",
                        help="comma-separated arms that never saw the labels")
    args = parser.parse_args(argv)

    store = ContextStore.load()
    arms = {}
    for spec in args.arm:
        name, _, path = spec.partition(":")
        p = Path(path)
        if p.is_file():
            arms[name] = {r["message_id"]: r for r in csv.DictReader(p.open(newline=""))}
    blind = {b.strip() for b in args.blind.split(",") if b.strip()}

    pairs = []
    for m in store.incoming:
        if not m.message_text:
            continue
        for r in store.samples:
            if r["user_id"] != m.user_id:
                continue
            if text_similarity(m.message_text, r["message_text"]) >= THRESHOLD:
                pairs.append((m.message_id, r["message_id"], r["action"], r["message_type"]))
                break

    print(f"  gold-transfer pairs (same recipient, similarity >= {THRESHOLD}): {len(pairs)}\n")
    if not pairs:
        return 0

    action_score = {n: 0 for n in arms}
    type_score = {n: 0 for n in arms}
    for mid, sid, gold_action, gold_type in pairs:
        for name, table in arms.items():
            row = table.get(mid)
            if row is None:
                continue
            action_score[name] += row["action"] == gold_action
            type_score[name] += row["message_type"] == gold_type
            if row["action"] != gold_action:
                print(f"  MISS  {name:9} {mid} -> {row['action']} (gold {gold_action}, from {sid})")

    print()
    for name in arms:
        tag = "blind" if name in blind else "saw labels during development"
        print(f"  {name:10} action {action_score[name]}/{len(pairs)} "
              f"({action_score[name]/len(pairs):.0%})   "
              f"type {type_score[name]}/{len(pairs)} ({type_score[name]/len(pairs):.0%})   [{tag}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
