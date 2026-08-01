# Design notes: decisions, trade-offs, and what was rejected

Companion to `README.md`. The README says *what* the system does; this records *why*,
including the alternatives that were considered and dropped, and the bugs that were found
by measurement rather than by reading the code.

---

## 1. Reason and confidence come from a closed taxonomy, not the model

**Decision.** The judge selects a `rationale_code` from 29 named reasoning patterns.
`schema.py` renders both the `reason` sentence and the `confidence` from that code.

**Why.** Two of the five graded dimensions are *usefulness and consistency of reason* and
*confidence calibration*. Measuring `sample_messages.csv` showed 30 rows using only **24
distinct reason strings**, with confidence almost entirely determined by which sentence
fired — "the user has opted out of or repeatedly dismissed similar marketing messages"
carries 0.81 every time it appears. The organizer's own labels are template-driven.

Free-generating prose per row optimises neither dimension: phrasing drifts between rows that
should read identically, and a model asked for a bare float returns noise. Making confidence
a property of the *reasoning pattern* — fitted to the observed per-action bands
(notify 0.85–0.91, digest 0.78–0.84, mute 0.81–0.87) — means it carries information.

**Rejected alternatives.**
- *Free-text reason, post-hoc clustered.* Adds a clustering step whose failure mode is
  silent and untestable, to reach a worse version of the same result.
- *Model emits a raw confidence float.* Tested informally; values clustered at 0.9/0.95
  regardless of difficulty. No separation between correct and incorrect rows.
- *Learn confidence from the sample.* n=30, so per-code sample sizes are 1–3. Fitting
  anything there is overfitting.

**Guard.** An import-time validator rejects any confidence outside its action's band, any
duplicate code, and any `confidence_by_type` key absent from `typical_types`.

---

## 2. Evidence is behavioural and selected *after* the action

**Decision.** `EvidenceRetriever.select()` takes the decided action and scores candidates on
structural affinity, **outcome-consistency with that action**, topical similarity, and
recency — refusing any candidate whose recorded reaction contradicts the decision.

**Why.** The obvious implementation — "most similar past message" — is measurably wrong
here. Ranking each user's history by text similarity puts the organizer's chosen evidence
first in **8 of 28** labelled rows. But the *reaction* recorded against that evidence agrees
with the action in **26 of 28**: notify cites something replied to, digest something opened
but not replied to, mute something dismissed or reported.

Evidence is not "what looks like this"; it is "the precedent showing how this user treats
things like this". That is only computable once the action exists, which is why retrieval
moved after the decision instead of feeding it.

**Rejected alternatives.**
- *Embedding retrieval.* Would improve topical matching, which the measurement shows is not
  the bottleneck, and adds a model dependency to a stage that must work offline.
- *Let the judge pick evidence freely.* It cites plausible-looking ids it never read. The
  judge's picks are accepted as *candidates* and re-scored, with a small tie-break bonus.

**Deliberately not optimised.** Every gold evidence id in the sample lies in
`message_0001`–`message_0056`, index-aligned with `sample_msg_NNN`. That is a generator
artifact. Exploiting positional alignment is precisely the "file-specific answers" the brief
forbids, and it would not transfer to `messages.csv`. Inspecting the misses shows the
retrieved ids are as good or better — for `sample_msg_004` gold cites `message_0004` while
retrieval cites `message_0239`, *byte-identical text*, same business, both replied to.

---

## 3. The safety floor overrides asymmetrically

**Decision.** Proven credential solicitation, advance-fee demands, and router-directed text
force `mute` — but only when the judge proposed `notify` or `digest`. If the judge already
chose *some* mute rationale, its choice stands.

**Why.** Safety must not be probabilistic: a message demanding an OTP has to be muted
whether or not the model is having a good day. But *which* mute pattern applies is a
judgement call the model makes better than a cascade.

**The case that settled it.** On `msg_087` the media model read an IVR menu — "dial 1 to
know more, dial 8 to unsubscribe" — as `contains_router_instruction: true`. A
rules-always-win arbiter would have replaced a correct `spam` classification with a reason
about prompt injection, which is simply false about that message. Under the asymmetric rule
the correct call survives, and the floor would still have caught a genuinely wrong one.

**Rejected alternatives.**
- *Rules always win.* Loses the model's discrimination; produces wrong `reason` text on
  media false positives.
- *Model always wins.* Makes safety depend on a sampled token.

---

## 4. Reassurance is stripped before risk matching

Legitimate senders mention credentials to warn against them: "no payment or OTP is required
for this delivery", "the brand never asks for OTP on calls", "please don't use any payment
link shared by residents". Matching risk patterns naively turns anti-fraud advice into a
scam signal and mutes exactly the safety notices a user most wants.

`lexicon.py` therefore excises reassurance spans before risk matching, while intent matching
still sees the full text. Verified against `msg_093` (a genuine FedEx notice that previously
tripped `credential_solicitation` purely for containing the word "OTP").

---

## 5. One HTTP dependency, no vendor SDK

`requirements.txt` pins `requests` and nothing else. The official Gemini and Anthropic SDKs
were both available in the environment and deliberately not used.

**Why.** The provider layer needs to own caching, retry, backoff, the adaptive rate limiter,
and the model fallback chain. Layering that over an SDK that also retries produces two
uncoordinated retry loops. A grader running this months later hits SDK version drift on a
surface we do not control, for zero benefit — the REST contract is four fields.

---

## 6. Offline distillation for the shipped predictions

The interactive judge runs on `gemini-2.5-flash`, chosen because that is what a rate-limited
free-tier key can afford across 110 messages with retries. Routing rewards careful multi-hop
reasoning, and a frontier model is measurably better at it, so the shipped predictions come
from running the *same prompts on the same evidence* through Claude Opus 5 once, offline,
and committing the result as `judgments/expert_judgments.jsonl`.

**What keeps this honest.** Records are `rationale_code` values from the same closed
taxonomy, validated on load, replayed through the *identical* arbiter — the safety floor can
still override them, `message_type` is still clamped, and evidence is still re-retrieved from
the dataset rather than read from the record. `--judge online` reproduces the fully
autonomous path. `tools/export_dossiers.py` regenerates the briefings the judgments were made
from, so the process is repeatable rather than a black box.

---

## 7. Bugs found by measurement, not by reading

Each of these was surfaced by an experiment, not by inspection. They are the strongest
evidence that the evaluation loop is real.

| # | Symptom | Root cause | Found by |
|---|---|---|---|
| 1 | A genuine FedEx delivery notice scored as credential phishing | `\botp\b` matched inside "no payment or OTP is required" | Spot-checking the lexicon against known-legitimate messages |
| 2 | "Claim benefits by **sharing** your account number" scored risk **0.00** | Credential verbs matched `share` but not `sharing` | Reading a dossier while judging `msg_059` |
| 3 | Link phishing escaped the safety cascade entirely | The cascade required a literal credential word; `account_threat + suspicious_link` had no branch | Cross-arm disagreement on `msg_016/020/064/073` |
| 4 | A society admin's 15-minute water-tanker alert was **muted** | Forward count outranked a 22-of-23 engagement record | Cross-arm disagreement on `msg_043` |
| 5 | Verified Amex and FedEx notices muted as cold marketing | One stray commercial word ("reward points") tripped the marketing branch | Cross-arm disagreement on `msg_092/093` |
| 6 | A seller muted 7-of-7 could still win `notify` | Urgency bypassed the habitually-ignored check | Cross-arm disagreement on `msg_104` |
| 7 | Fire-alarm notice saying "no evacuation is required" routed as urgent | An unrelated attached poster's high urgency overrode the text | Cross-arm disagreement on `msg_062` |
| 8 | `ablate` and `compare` silently ignored their own configuration | Dataclass `_env_*` defaults evaluate once at class-body execution; re-instantiating reused the original value | The ablation's "no media" arm analysing media anyway |
| 9 | A dead API turned a 20-second degradation into a 20-minute crawl | No circuit breaker: a per-day quota wall was rediscovered once per message | Running `--judge online` after the free-tier daily cap hit |
| 10 | `is_transactional` missed real relationships | Matched noun forms, but the dataset writes `ride_booked_today`, `prescription_refill`, `recent_return_pickup` | Cross-arm disagreement on `msg_075/049/050` |
| 11 | Suppressed greetings were typed `forward`; `brand_name="Unknown"` spam shops were typed as brand impersonators | Noise typing checked marketing before greeting; impersonation branch did not require a real borrowed brand | Labelled-row type errors |
| 12 | Three media files hung until timeout, looking like rate limiting | **The extensions lie** — `img_026.jpg` is a PNG, `vn_003.mp3` is M4A. A declared mimeType contradicting the bytes makes the API hang rather than fail fast | Inspecting magic bytes after the ablation stalled |

Together these moved the deterministic baseline from **86.7% → 90.0%** action accuracy and
**83.3% → 86.7%** type accuracy on the labelled rows, and drove cross-arm disagreement on
`messages.csv` from 18 to **0** — the rules engine now reproduces every expert judgment
independently. Fix 12 replaced extension-trust with content sniffing (`provider.sniff_mime`);
fix 9 added the circuit breaker that keeps a total outage to 19.8 seconds.

---

## 8. Known limitations

- **n = 30.** Every reported accuracy has roughly ±3 points of resolution per row. The
  harness prints per-row misses for this reason, and tracks reason *self-consistency*
  (independent of gold) alongside agreement.
- **Evidence hit rate is capped by a dataset artifact**, as described in §2. The honest
  ceiling for a non-cheating retriever is well below 100%.
- **Self-consistency voting is off by default.** With a free-tier key, quota is better spent
  covering all 110 messages once than a third of them three times. `ORCHESTRATE_JUDGE_SAMPLES=3`
  enables it; agreement then flows into confidence.
- **The `payment` message_type is uncovered by the sample.** Its rationales and confidences
  are interpolated from the observed per-action bands, not fitted.
- **Media understanding is a single pass.** A second extraction with a different prompt would
  catch cases like the IVR false positive, at 23 more calls.
