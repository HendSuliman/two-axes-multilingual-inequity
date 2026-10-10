#!/usr/bin/env python3
"""Downstream zero-shot cross-lingual transfer: Meccan vs Medinan verse classification.

Train a logistic-regression probe on English verse embeddings, evaluate zero-shot on every
other language's embeddings of the SAME held-out verses (labels transfer because the corpus
is verse-aligned). Then test which inequity axis predicts the per-language transfer score.

Usage (from anywhere):
    python code/downstream_transfer.py                         # multilingual-E5, embeds everything
    python code/downstream_transfer.py BAAI/bge-m3             # another encoder
    python code/downstream_transfer.py --analysis-only         # skip embedding; analyse the
                                                               # shipped results for all 4 encoders

Correction relative to the August release: the per-encoder correlations now control for
THAT ENCODER'S OWN retrieval accuracy (results/retrieval_per_language.csv). The earlier
version read the single `retrieval` column of joint_language_table.csv, which is the mean
pooled over all four encoders, so every encoder was compared against the same vector.
"""
import io, json, re, sys
import numpy as np, pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import METADATA, RESULTS, FERTILITY, corpus_dir, outputs

SEED = 42
N_EVAL = 1000
N_TRAIN_EN = 2000
MEDINAN = {2,3,4,5,8,9,13,22,24,33,47,48,49,55,57,58,59,60,61,62,63,64,65,66,76,98,99,110}
TAG_OF = {"sentence-transformers/LaBSE": "labse", "BAAI/bge-m3": "bgem3",
          "intfloat/multilingual-e5-base": "e5",
          "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": "mpnet"}
SECOND_BOSNIAN = "bosnian-mihanovich"      # one Bosnian per language for n=40 correlations


def lang_key(fname):
    if fname.startswith("bosnian_mihanovich"): return "bosnian-mihanovich"
    if fname.startswith("bosnian_rwwad"): return "bosnian-rwwad"
    return fname.split("_")[0]


def read_corpus_csv(path):
    txt = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    m = re.search(r"^id,sura,aya", txt, flags=re.M)          # skip the provenance preamble
    df = pd.read_csv(io.StringIO(txt[m.start():]))[["sura", "aya", "translation"]].dropna()
    df["sura"] = df.sura.astype(int); df["aya"] = df.aya.astype(int)
    return df


def run_probe(model_name):
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    rng = np.random.default_rng(SEED)
    langs = {lang_key(f.name): f for f in sorted(corpus_dir().glob("*.csv"))}
    print(f"{len(langs)} translation files")
    en = read_corpus_csv(langs["english"])
    keys = list(en[["sura", "aya"]].itertuples(index=False, name=None))
    assert len(keys) == 6236
    idx = rng.permutation(len(keys))
    eval_keys = [keys[i] for i in sorted(idx[:N_EVAL])]
    train_keys = [keys[i] for i in sorted(idx[N_EVAL:N_EVAL + N_TRAIN_EN])]
    lab = lambda ks: np.array([1 if s in MEDINAN else 0 for s, a in ks])
    y_eval, y_train = lab(eval_keys), lab(train_keys)

    model = SentenceTransformer(model_name, device="cpu")
    prefix = "query: " if "e5" in model_name else ""
    embed = lambda ts: model.encode([prefix + t for t in ts], batch_size=64, convert_to_numpy=True,
                                    normalize_embeddings=True, show_progress_bar=False)
    pick = lambda df, ks: [str(df.set_index(["sura", "aya"])["translation"].get(k, "")) for k in ks]

    clf = LogisticRegression(max_iter=3000).fit(embed(pick(en, train_keys)), y_train)
    rows = []
    for lang, f in sorted(langs.items()):
        pred = clf.predict(embed(pick(read_corpus_csv(f), eval_keys)))
        rows.append({"encoder": TAG_OF.get(model_name, model_name.split("/")[-1]), "language": lang,
                     "accuracy": round(float(accuracy_score(y_eval, pred)), 4),
                     "macro_f1": round(float(f1_score(y_eval, pred, average="macro")), 4)})
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def partial_spearman(x, y, z):
    """Rank partial correlation of x and y controlling z."""
    from scipy.stats import rankdata, pearsonr
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    Z = np.column_stack([np.ones(len(rz)), rz])
    res = lambda a: a - Z @ np.linalg.lstsq(Z, a, rcond=None)[0]
    return pearsonr(res(rx), res(ry))[0]


def analyse(down):
    """Correlate each encoder's transfer with its OWN retrieval accuracy and with GPT-4o fertility."""
    from scipy.stats import spearmanr
    retr = pd.read_csv(RESULTS / "retrieval_per_language.csv")
    own = (retr.assign(encoder=retr.model.map(TAG_OF)).dropna(subset=["encoder"])
               [["encoder", "language", "mean_top1"]].rename(columns={"mean_top1": "retrieval_own"}))
    fert = pd.read_csv(FERTILITY)[["language", "gpt4o_o200k_tok_per_verse"]]
    joshi = pd.read_csv(METADATA / "joint_language_table.csv")[["language", "joshi"]]
    res = {}
    for enc, d in down.groupby("encoder"):
        m = (d.merge(own, on=["encoder", "language"]).merge(fert, on="language")
              .merge(joshi, on="language"))
        m = m[m.language != SECOND_BOSNIAN]
        t = m.macro_f1.values
        r = {"n": int(len(m))}
        for axis, col in [("retrieval_own", "retrieval_own"), ("fertility_gpt4o", "gpt4o_o200k_tok_per_verse"),
                          ("joshi", "joshi")]:
            rho, p = spearmanr(t, m[col])
            r[axis] = {"rho": round(float(rho), 3), "p": float(p)}
        r["fertility_given_retrieval"] = round(float(partial_spearman(
            t, m.gpt4o_o200k_tok_per_verse.values, m.retrieval_own.values)), 3)
        res[enc] = r
        print(f"{enc:6s} n={r['n']}  rho(transfer, own retrieval)={r['retrieval_own']['rho']:+.3f}  "
              f"rho(transfer, fertility)={r['fertility_gpt4o']['rho']:+.3f}  "
              f"partial(fertility | retrieval)={r['fertility_given_retrieval']:+.3f}")
    return res


def main():
    args = sys.argv[1:]
    out = outputs()
    if "--analysis-only" in args:
        down = pd.read_csv(RESULTS / "downstream_all_encoders.csv")
        tag = "all_encoders"
    else:
        model_name = args[0] if args else "intfloat/multilingual-e5-base"
        down = run_probe(model_name)
        tag = TAG_OF.get(model_name, model_name.split("/")[-1])
        down.to_csv(out / f"downstream_{tag}.csv", index=False)
    res = analyse(down)
    json.dump(res, open(out / f"downstream_{tag}_correlations.json", "w"), indent=1)
    print("wrote", out / f"downstream_{tag}_correlations.json")


if __name__ == "__main__":
    main()
