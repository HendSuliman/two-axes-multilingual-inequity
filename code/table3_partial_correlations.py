#!/usr/bin/env python3
"""Table 4 (per-encoder downstream transfer; Table 3 in August), end to end from the shipped results.

    python code/table3_partial_correlations.py

For each encoder: transfer macro-F1 against that encoder's OWN retrieval accuracy and against
GPT-4o fertility, and the rank partial correlation of transfer with fertility controlling for
retrieval, with a 10,000-sample bootstrap interval. n = 40 (one Bosnian translation).

The August submission controlled for the single `retrieval` column of joint_language_table.csv,
which is the mean pooled over all four encoders. The POOLED columns printed alongside apply that
pooled control to the current results, for comparison. (The August partial-correlation column had no
surviving code, and the August retrieval scores for BGE-M3, multilingual-E5 and MPNet were later
re-run; see the README.) The own-encoder columns are the values reported in the paper.
Writes outputs/table_downstream.tex and outputs/table3_partial_correlations.csv.
"""
import sys
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, rankdata, norm, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import METADATA, RESULTS, FERTILITY, outputs

ENC = {"sentence-transformers/LaBSE": "labse", "BAAI/bge-m3": "bgem3",
       "intfloat/multilingual-e5-base": "e5",
       "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": "mpnet"}
NAME = {"labse": "LaBSE", "bgem3": "BGE-M3", "mpnet": "MPNet", "e5": "mult.-E5"}
FLOOR = 0.4318          # exact majority-class macro-F1 on the 1,000 held-out verses
N_BOOT = 10000


def partial_spearman(x, y, z):
    """Rank partial correlation of x and y controlling z (rank-residual regression)."""
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    Z = np.column_stack([np.ones(len(rz)), rz])
    res = lambda a: a - Z @ np.linalg.lstsq(Z, a, rcond=None)[0]
    return pearsonr(res(rx), res(ry))[0]


def main():
    down  = pd.read_csv(RESULTS / "downstream_all_encoders.csv")
    joint = pd.read_csv(METADATA / "joint_language_table.csv")        # `retrieval` = pooled mean
    retr  = pd.read_csv(RESULTS / "retrieval_per_language.csv")
    fert  = pd.read_csv(FERTILITY)
    per_enc = (retr.assign(e=retr.model.map(ENC)).dropna(subset=["e"])
                   .pivot(index="language", columns="e", values="mean_top1"))
    base = (joint.merge(per_enc, left_on="language", right_index=True)
                 .merge(fert[["language", "gpt4o_o200k_tok_per_verse"]], on="language"))
    base = base[base.language != "bosnian-mihanovich"]

    rng = np.random.default_rng(42)
    rows = []
    for enc in ["labse", "bgem3", "mpnet", "e5"]:
        s = base.merge(down[down.encoder == enc][["language", "macro_f1"]], on="language")
        t, f = s.macro_f1.values, s.gpt4o_o200k_tok_per_verse.values
        r_own, r_pool = s[enc].values, s.retrieval.values
        boot = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(s), len(s))
            try: boot.append(partial_spearman(t[i], f[i], r_own[i]))
            except Exception: pass
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"encoder": NAME[enc], "n": len(s), "mean": round(t.mean(), 3),
                     "min": round(t.min(), 3), "floor": int((np.abs(t - FLOOR) < 1e-9).sum()),
                     "rho_ret": round(spearmanr(t, r_own)[0], 3), "rho_fert": round(spearmanr(t, f)[0], 3),
                     "partial": round(partial_spearman(t, f, r_own), 3),
                     "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
                     "rho_ret_POOLED": round(spearmanr(t, r_pool)[0], 3),
                     "partial_POOLED": round(partial_spearman(t, f, r_pool), 3)})
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))
    print("\nPOOLED columns use the August (pooled) control; own-encoder columns are the corrected values.")
    n = int(tab.n.iloc[0])
    print(f"Smallest |partial rho| detectable at n={n}, alpha=0.05: {np.tanh(norm.ppf(0.975)/np.sqrt(n-4)):.3f}")

    out = outputs()
    tab.to_csv(out / "table3_partial_correlations.csv", index=False)
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Encoder & Mean & Min & Floor & $\rho_{\mathrm{ret}}$ & $\rho_{\mathrm{fert}}$ & "
             r"$\rho_{\mathrm{fert}\mid\mathrm{ret}}$ & 95\% CI \\", r"\midrule"]
    for _, r in tab.iterrows():
        lines.append(f"{r.encoder} & {r['mean']:.2f} & {r['min']:.2f} & {r.floor} & ${r.rho_ret:+.2f}$ & "
                     f"${r.rho_fert:+.2f}$ & ${r.partial:+.2f}$ & $[{r.ci_lo:+.2f},{r.ci_hi:+.2f}]$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Zero-shot Meccan/Medinan transfer per encoder, with the rank partial correlation "
              r"controlling for \emph{that encoder's own} retrieval accuracy and a 10{,}000-sample "
              r"bootstrap interval.}", r"\label{tab:downstream}", r"\end{table}"]
    (out / "table_downstream.tex").write_text("\n".join(lines))
    print("wrote", out / "table_downstream.tex")


if __name__ == "__main__":
    main()
