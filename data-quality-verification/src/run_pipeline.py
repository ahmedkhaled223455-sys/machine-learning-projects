"""Run the full verification pipeline and score it against ground truth.

Routing policy (the human-in-the-loop design):
  auto-reject : any hard rule violation, near-exact duplicate, or
                extreme text-quality score
  review      : soft flags — suspicious but not certain (borderline
                perplexity, moderate similarity, high repetition)
  auto-accept : everything else

The pipeline never sees ground-truth labels; reference statistics
(price ranges, the text LM) are fit on records that pass the rule
layer — a bootstrapping choice that mirrors reality, where a trusted
seed sample exists but exhaustive labels do not. Ground truth is used
only for evaluation.

Run:  python src/generate_dataset.py && python src/run_pipeline.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from checks import (MinHashDeduper, PriceRangeCheck, TextQualityModel,
                    rule_checks)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    RESULTS.mkdir(exist_ok=True)
    records = load_jsonl(DATA / "records.jsonl")
    truth = {t["record_id"]: t["defects"]
             for t in load_jsonl(DATA / "ground_truth.jsonl")}
    print(f"{len(records)} records loaded")

    # ---- layer 1: rules ---------------------------------------------------
    flags = defaultdict(list)
    for rec in records:
        flags[rec["record_id"]].extend(rule_checks(rec))

    rule_clean = [r for r in records if not flags[r["record_id"]]]
    print(f"rule layer: {len(records) - len(rule_clean)} records flagged; "
          f"{len(rule_clean)} pass to reference pool")

    # ---- layer 2: statistical ranges (fit on rule-clean pool) ------------
    price_check = PriceRangeCheck().fit(rule_clean)
    for rec in records:
        flags[rec["record_id"]].extend(price_check.check(rec))

    # ---- layer 3a: near-duplicate detection ------------------------------
    ids = [r["record_id"] for r in records]
    texts = [str(r.get("description") or "") for r in records]
    dup_flags = MinHashDeduper().find_duplicates(ids, texts)
    for rid, flag in dup_flags.items():
        flags[rid].append(flag)

    # ---- layer 3b: text quality (LM fit on rule-clean descriptions) ------
    tq = TextQualityModel().fit(
        [str(r["description"]) for r in rule_clean])
    ppl = {r["record_id"]: tq.perplexity(str(r.get("description") or ""))
           for r in records}
    rep = {r["record_id"]: tq.repetition_ratio(str(r.get("description") or ""))
           for r in records}
    # Robust thresholds (median + k*MAD in log space): the reference pool
    # still contains undetected gibberish, so naive percentiles inflate
    # the bar — the same poisoning problem as price ranges.
    log_ppl = np.log([ppl[r["record_id"]] for r in rule_clean])
    med = float(np.median(log_ppl))
    mad = float(np.median(np.abs(log_ppl - med))) + 1e-9
    ppl_hi = float(np.exp(med + 10 * mad))                 # gibberish bar
    ppl_soft = float(np.exp(med + 6 * mad))
    for r in records:
        rid = r["record_id"]
        if ppl[rid] > ppl_hi:
            flags[rid].append(("gibberish_text", "hard", f"ppl {ppl[rid]:.0f}"))
        elif ppl[rid] > ppl_soft:
            flags[rid].append(("odd_text", "soft", f"ppl {ppl[rid]:.0f}"))
        if rep[rid] > 0.65:
            flags[rid].append(("repeated_boilerplate", "hard",
                               f"repetition {rep[rid]:.2f}"))
        elif rep[rid] > 0.5:
            flags[rid].append(("repetitive_text", "soft",
                               f"repetition {rep[rid]:.2f}"))

    # ---- routing ----------------------------------------------------------
    routing = {}
    for r in records:
        rid = r["record_id"]
        sev = [s for _, s, _ in flags[rid]]
        routing[rid] = ("auto_reject" if "hard" in sev
                        else "review" if "soft" in sev
                        else "auto_accept")
    route_counts = Counter(routing.values())
    print("routing:", dict(route_counts))

    # ---- evaluation against ground truth ---------------------------------
    is_bad = {rid: bool(d) for rid, d in truth.items()}
    flagged = {rid: routing[rid] != "auto_accept" for rid in routing}
    tp = sum(1 for rid in routing if flagged[rid] and is_bad[rid])
    fp = sum(1 for rid in routing if flagged[rid] and not is_bad[rid])
    fn = sum(1 for rid in routing if not flagged[rid] and is_bad[rid])
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accepted = [rid for rid in routing if routing[rid] == "auto_accept"]
    leakage = sum(1 for rid in accepted if is_bad[rid]) / len(accepted)
    review_load = route_counts["review"] / len(records)
    print(f"flagged-vs-corrupted precision {precision:.3f} "
          f"recall {recall:.3f}")
    print(f"defect leakage into auto-accept: {leakage:.4f} | "
          f"human review load: {review_load:.1%}")

    # per-failure-mode recall
    mode_recall = {}
    for mode in sorted({m for d in truth.values() for m in d}):
        rids = [rid for rid, d in truth.items() if mode in d]
        caught = sum(1 for rid in rids if flagged[rid])
        mode_recall[mode] = round(caught / len(rids), 3)
    print("\nDetection recall by failure mode:")
    for mode, rc in sorted(mode_recall.items(), key=lambda t: t[1]):
        print(f"  {mode:22s} {rc:.1%}")

    # ---- review-queue examples (the manual-review artifact) --------------
    lines = ["# Review queue sample", ""]
    review_ids = [rid for rid in routing if routing[rid] == "review"][:8]
    rec_by_id = {r["record_id"]: r for r in records}
    for rid in review_ids:
        reasons = "; ".join(f"{n} ({s}: {d})" for n, s, d in flags[rid])
        lines += [f"## {rid}", f"- reasons: {reasons}",
                  f"- truth: {truth[rid] or 'clean'}",
                  f"- description: `{str(rec_by_id[rid].get('description'))[:120]}`",
                  ""]
    (RESULTS / "review_queue_sample.md").write_text("\n".join(lines))

    # ---- figures ----------------------------------------------------------
    modes, recalls = zip(*sorted(mode_recall.items(), key=lambda t: t[1]))
    plt.figure(figsize=(8, 4.5))
    plt.barh(modes, recalls, color="#31708E")
    plt.xlabel("detection recall")
    plt.title("Recall by injected failure mode")
    plt.xlim(0, 1.05)
    plt.tight_layout()
    plt.savefig(RESULTS / "recall_by_mode.png", dpi=150)
    plt.close()

    bad_ppl = [np.log(ppl[rid]) for rid in ppl if is_bad[rid]]
    ok_ppl = [np.log(ppl[rid]) for rid in ppl if not is_bad[rid]]
    plt.figure(figsize=(7, 4))
    plt.hist(ok_ppl, bins=40, alpha=0.6, density=True, label="clean",
             color="#8896AB")
    plt.hist(bad_ppl, bins=40, alpha=0.6, density=True, label="corrupted",
             color="#B1624E")
    plt.axvline(np.log(ppl_hi), color="k", ls=":", lw=1,
                label="auto-reject bar")
    plt.xlabel("log perplexity of description (char trigram LM)")
    plt.legend()
    plt.title("LM-based text-quality signal")
    plt.tight_layout()
    plt.savefig(RESULTS / "perplexity_signal.png", dpi=150)
    plt.close()

    with open(RESULTS / "metrics.json", "w") as f:
        json.dump({"routing": dict(route_counts),
                   "precision_flagged": round(precision, 4),
                   "recall_flagged": round(recall, 4),
                   "leakage_into_accept": round(leakage, 4),
                   "review_load": round(review_load, 4),
                   "recall_by_mode": mode_recall}, f, indent=2)
    print(f"\nSaved metrics, figures, review-queue sample to {RESULTS}")

    # ---- optional MLflow experiment tracking ------------------------------
    # If mlflow is installed (pip install mlflow), each pipeline run is
    # logged with its thresholds, headline metrics, and artifacts so runs
    # can be compared in the MLflow UI (`mlflow ui`, http://localhost:5000).
    try:
        import mlflow
    except ImportError:
        print("mlflow not installed -- skipping experiment tracking")
    else:
        mlflow.set_experiment("data-quality-verification")
        with mlflow.start_run():
            mlflow.log_params({
                "n_records": len(records),
                "ppl_hard_threshold": round(ppl_hi, 2),
                "ppl_soft_threshold": round(ppl_soft, 2),
                "repetition_hard": 0.65,
                "repetition_soft": 0.5,
            })
            mlflow.log_metrics({
                "precision_flagged": precision,
                "recall_flagged": recall,
                "leakage_into_accept": leakage,
                "review_load": review_load,
            })
            mlflow.log_artifacts(str(RESULTS))
        print("logged run to MLflow experiment 'data-quality-verification'")


if __name__ == "__main__":
    main()
