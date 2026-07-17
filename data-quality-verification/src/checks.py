"""Quality-check engine: rule-based, statistical, and AI-driven layers.

Layer 1 — deterministic rules (cheap, precise):
    schema/type validation, canonical category matching, timestamp
    window, mojibake regex.
Layer 2 — statistical checks (learned from a reference sample):
    per-category robust price ranges.
Layer 3 — fuzzy/AI checks (catch what rules cannot):
    - near-duplicate detection with MinHash over character shingles
    - text-quality scoring with a character trigram language model
      (records whose descriptions have abnormally high perplexity are
      gibberish; abnormally low repetition entropy means boilerplate).
      This is the same perplexity-filtering idea used to clean LLM
      pretraining corpora.

Every check returns (flag_name, severity, detail); the pipeline
aggregates them into a routing decision.
"""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

CANONICAL_CATEGORIES = {"electronics", "home", "outdoors", "toys",
                        "office", "grocery"}
TS_WINDOW = (1_735_689_600, 1_751_241_600)          # collection window
MOJIBAKE = re.compile(r"(Ã.|â€.|�)")


# ---------------------------------------------------------------------------
# Layer 1: rules
# ---------------------------------------------------------------------------
def rule_checks(rec: dict) -> list[tuple[str, str, str]]:
    flags = []
    for field in ["category", "price_usd", "timestamp", "description"]:
        if rec.get(field) is None:
            flags.append(("missing_field", "hard", field))
    price = rec.get("price_usd")
    if price is not None and not isinstance(price, (int, float)):
        flags.append(("wrong_type", "hard", f"price_usd={price!r}"))
    cat = rec.get("category")
    if cat is not None and cat not in CANONICAL_CATEGORIES:
        norm = str(cat).strip().lower().rstrip("s")
        fixable = any(norm == c.rstrip("s") for c in CANONICAL_CATEGORIES)
        flags.append(("category_typo" if fixable else "unknown_category",
                      "hard", str(cat)))
    ts = rec.get("timestamp")
    if isinstance(ts, int) and not (TS_WINDOW[0] <= ts <= TS_WINDOW[1]):
        flags.append(("stale_timestamp", "hard", str(ts)))
    desc = rec.get("description")
    if desc and MOJIBAKE.search(desc):
        flags.append(("encoding_artifact", "hard",
                      MOJIBAKE.search(desc).group()))
    return flags


# ---------------------------------------------------------------------------
# Layer 2: statistical range check (fit on reference records)
# ---------------------------------------------------------------------------
class PriceRangeCheck:
    def fit(self, records: list[dict]):
        by_cat = defaultdict(list)
        for r in records:
            if isinstance(r.get("price_usd"), (int, float)) \
                    and r.get("category") in CANONICAL_CATEGORIES:
                by_cat[r["category"]].append(float(r["price_usd"]))
        # IQR fences, not percentiles: the reference pool inevitably
        # contains undetected defects, and raw percentiles get poisoned
        # by exactly the outliers we are trying to catch. Quartiles are
        # robust to a few percent contamination.
        self.bounds = {}
        for cat, prices in by_cat.items():
            q1, q3 = np.percentile(prices, [25, 75])
            iqr = q3 - q1
            self.bounds[cat] = (max(0.01, q1 - 3 * iqr), q3 + 3 * iqr)
        return self

    def check(self, rec: dict) -> list[tuple[str, str, str]]:
        price, cat = rec.get("price_usd"), rec.get("category")
        if isinstance(price, (int, float)) and cat in self.bounds:
            lo, hi = self.bounds[cat]
            if not (lo <= float(price) <= hi):
                return [("out_of_range", "hard", f"{cat}: {price}")]
        return []


# ---------------------------------------------------------------------------
# Layer 3a: near-duplicate detection (MinHash)
# ---------------------------------------------------------------------------
class MinHashDeduper:
    """MinHash signatures over character 3-gram shingles; records whose
    signatures agree on >= `sim_threshold` of hash functions are
    duplicate candidates (verified with exact Jaccard)."""

    def __init__(self, num_hashes: int = 64, shingle: int = 3,
                 sim_threshold: float = 0.8, seed: int = 0):
        self.H = num_hashes
        self.k = shingle
        self.thr = sim_threshold
        rng = np.random.default_rng(seed)
        self.prime = 2_147_483_647
        self.a = rng.integers(1, self.prime, self.H)
        self.b = rng.integers(0, self.prime, self.H)

    def _shingles(self, text: str) -> set[int]:
        t = " ".join(str(text).lower().split())
        return {hash(t[i:i + self.k]) & 0x7FFFFFFF
                for i in range(max(1, len(t) - self.k + 1))}

    def _signature(self, sh: set[int]) -> np.ndarray:
        vals = np.fromiter(sh, dtype=np.int64)
        return ((self.a[:, None] * vals[None, :] + self.b[:, None])
                % self.prime).min(axis=1)

    def find_duplicates(self, ids: list[str], texts: list[str]):
        shingle_sets = [self._shingles(t) for t in texts]
        sigs = np.stack([self._signature(s) for s in shingle_sets])
        # Band the signatures (8 bands x 8 rows) into candidate buckets.
        buckets = defaultdict(list)
        for i in range(len(ids)):
            for band in range(8):
                key = (band, hash(sigs[i, band * 8:(band + 1) * 8].tobytes()))
                buckets[key].append(i)
        # A duplicate pair is symmetric: without provenance metadata there
        # is no way to know which member is the original, so BOTH are
        # quarantined for review (auto-rejecting one on a coin flip would
        # discard good data half the time).
        flagged = {}
        for _, members in buckets.items():
            if len(members) < 2:
                continue
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    i, j = members[x], members[y]
                    inter = len(shingle_sets[i] & shingle_sets[j])
                    union = len(shingle_sets[i] | shingle_sets[j])
                    jac = inter / union if union else 0.0
                    if jac >= self.thr:
                        for a, b in [(i, j), (j, i)]:
                            prev = flagged.get(ids[a])
                            if prev is None or jac > prev[0]:
                                flagged[ids[a]] = (jac, ids[b])
        return {rid: ("near_duplicate", "soft",
                      f"jaccard {sim:.2f} vs {other}")
                for rid, (sim, other) in flagged.items()}


# ---------------------------------------------------------------------------
# Layer 3b: text quality via character trigram perplexity + repetition
# ---------------------------------------------------------------------------
class TextQualityModel:
    def __init__(self, k_smooth: float = 0.2):
        self.k = k_smooth

    def fit(self, texts: list[str]):
        self.counts = defaultdict(lambda: defaultdict(float))
        self.vocab = set()
        for t in texts:
            t = f"^^{t.lower()}$"
            for i in range(2, len(t)):
                self.counts[t[i - 2:i]][t[i]] += 1
                self.vocab.add(t[i])
        self.V = len(self.vocab)
        return self

    def perplexity(self, text: str) -> float:
        t = f"^^{str(text).lower()}$"
        nll, n = 0.0, 0
        for i in range(2, len(t)):
            ctx = self.counts.get(t[i - 2:i])
            if ctx is None:
                p = 1.0 / self.V
            else:
                total = sum(ctx.values())
                p = (ctx.get(t[i], 0.0) + self.k) / (total + self.k * self.V)
            nll -= np.log(p)
            n += 1
        return float(np.exp(nll / max(n, 1)))

    @staticmethod
    def repetition_ratio(text: str) -> float:
        words = str(text).lower().split()
        if not words:
            return 1.0
        return 1.0 - len(set(words)) / len(words)
