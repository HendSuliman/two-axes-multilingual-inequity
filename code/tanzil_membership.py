#!/usr/bin/env python3
"""Check 1 of Appendix D (training exposure): does membership in Tanzil/OPUS explain retrieval beyond
Joshi resource tier?

    python code/tanzil_membership.py          # offline, CPU, about 1 minute

Membership comes from OPUS's info.yaml for Tanzil v1 (42 languages); 17 of the corpus's 40 languages
are absent. Freedman-Lane permutation test of the membership term controlling tier (20,000
permutations), on raw and rank-transformed retrieval, pooled and per encoder; bootstrap interval for
the pooled coefficient; two sensitivity analyses. n = 40 (bosnian-rwwad). Writes
outputs/tanzil_membership_regression.csv.
"""
import sys
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import METADATA, RESULTS, outputs

ABSENT = {"asante", "assamese", "croatian", "gujarati", "kannada", "khmer", "kyrgyz", "lingala", "lithuanian",
          "macedonian", "moore", "oromo", "pashto", "serbian", "tagalog", "vietnamese", "yoruba"}
ENC = {"sentence-transformers/LaBSE": "labse", "BAAI/bge-m3": "bgem3", "intfloat/multilingual-e5-base": "e5",
       "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": "mpnet"}
AFRICAN = {"moore", "asante", "oromo", "lingala", "yoruba", "hausa"}
rng = np.random.default_rng(42)


def ols(y, X):
    X = np.column_stack([np.ones(len(y)), X]); b, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ b
    return 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean())), b, X @ b, r


def freedman_lane(y, tier, m, n=20000):
    r2r, _, fit, res = ols(y, tier[:, None]); r2f, b, _, _ = ols(y, np.column_stack([tier, m])); d = r2f - r2r
    null = np.empty(n)
    for i in range(n):
        ys = fit + res[rng.permutation(len(y))]
        null[i] = ols(ys, np.column_stack([tier, m]))[0] - ols(ys, tier[:, None])[0]
    return r2r, d, b[2], float((null >= d).mean())


def main():
    J = pd.read_csv(METADATA / "joint_language_table.csv"); J = J[J.language != "bosnian-mihanovich"]
    R = pd.read_csv(RESULTS / "retrieval_per_language.csv")
    P = R.assign(e=R.model.map(ENC)).dropna(subset=["e"]).pivot(index="language", columns="e", values="mean_top1")
    D = J.set_index("language").join(P)
    D["pooled"] = D[["labse", "bgem3", "e5", "mpnet"]].mean(axis=1)
    D["tanzil"] = [0.0 if l in ABSENT else 1.0 for l in D.index]
    assert len(D) == 40 and D.tanzil.sum() == 23
    print(f"membership vs tier: Spearman {stats.spearmanr(D.tanzil, D.joshi)[0]:.2f}")
    rows = []
    def add(label, d, col):
        y, t, m = d[col].values, d.joshi.values.astype(float), d.tanzil.values
        r2r, dr2, coef, p = freedman_lane(y, t, m)
        _, dr2r, _, pr = freedman_lane(stats.rankdata(y), stats.rankdata(t), m, 5000)
        rows.append({"set": label, "retrieval": col, "n": len(d), "members": int(m.sum()), "R2_tier": round(r2r, 3),
                     "dR2_tanzil": round(dr2, 3), "coef_tanzil": round(coef, 3), "p_perm": round(p, 4),
                     "dR2_rank": round(dr2r, 3), "p_rank": round(pr, 4)})
    for col in ["pooled", "labse", "bgem3", "e5", "mpnet"]:
        add("all 40", D, col)
    add("without the six African languages", D[~D.index.isin(AFRICAN)], "pooled")
    D2 = D.copy(); D2.loc["kurdish", "tanzil"] = 0.0
    add("Kurdish counted as absent", D2, "pooled")
    T = pd.DataFrame(rows)
    g = np.random.default_rng(7); y, t, m = D.pooled.values, D.joshi.values.astype(float), D.tanzil.values; bs = []
    for _ in range(10000):
        i = g.integers(0, 40, 40)
        if m[i].min() != m[i].max(): bs.append(ols(y[i], np.column_stack([t[i], m[i]]))[1][2])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(T.to_string(index=False))
    print(f"pooled coefficient 95% bootstrap CI [{lo:.3f}, {hi:.3f}]")
    floor = ["moore", "asante", "oromo", "kurdish", "lingala"]
    rest = D[(D.joshi <= 1) & ~D.index.isin(floor)].pooled.mean()
    print(f"floor gap: tier 0-1 non-floor mean {rest:.3f} vs floor mean {D.loc[floor, 'pooled'].mean():.3f}")
    T.to_csv(outputs() / "tanzil_membership_regression.csv", index=False)


if __name__ == "__main__":
    main()
