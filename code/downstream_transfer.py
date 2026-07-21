#!/usr/bin/env python3
"""Downstream zero-shot cross-lingual transfer: Meccan vs Medinan verse classification.

Train a logistic-regression probe on English verse embeddings, evaluate zero-shot
on every other language's embeddings of the SAME held-out verses (labels transfer
because the corpus is verse-aligned). Then test which inequity axis predicts the
per-language transfer gap: retrieval accuracy (encoder axis) or tokenizer fertility.
"""
import io, json, sys, re
import numpy as np, pandas as pd
from pathlib import Path

SEED = 42
N_EVAL = 1000
N_TRAIN_EN = 2000
MODEL = sys.argv[1] if len(sys.argv) > 1 else "intfloat/multilingual-e5-base"
TAG = MODEL.split("/")[-1]
CORPUS = Path("corpus")

MEDINAN = {2,3,4,5,8,9,13,22,24,33,47,48,49,55,57,58,59,60,61,62,63,64,65,66,76,98,99,110}

def lang_key(fname):
    stem = fname.split("_")[0]
    if fname.startswith("bosnian_mihanovich"): return "bosnian-mihanovich"
    if fname.startswith("bosnian_rwwad"): return "bosnian-rwwad"
    return stem

def read_corpus_csv(path):
    txt = path.read_text(encoding="utf-8-sig", errors="replace")
    # skip provenance preamble; real header line starts with id,sura,aya
    m = re.search(r"^id,sura,aya", txt, flags=re.M)
    df = pd.read_csv(io.StringIO(txt[m.start():]))
    df = df[["sura", "aya", "translation"]].dropna()
    df["sura"] = df.sura.astype(int); df["aya"] = df.aya.astype(int)
    return df

def main():
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from scipy.stats import spearmanr

    rng = np.random.default_rng(SEED)
    files = sorted(CORPUS.glob("*.csv"))
    langs = {lang_key(f.name): f for f in files}
    print(f"{len(langs)} languages")

    en = read_corpus_csv(langs["english"])
    keys = list(en[["sura", "aya"]].itertuples(index=False, name=None))
    assert len(keys) == 6236
    idx = rng.permutation(len(keys))
    eval_idx = set(idx[:N_EVAL]); train_idx = set(idx[N_EVAL:N_EVAL + N_TRAIN_EN])
    eval_keys = [keys[i] for i in sorted(eval_idx)]
    train_keys = [keys[i] for i in sorted(train_idx)]
    y = lambda ks: np.array([1 if s in MEDINAN else 0 for s, a in ks])
    y_eval, y_train = y(eval_keys), y(train_keys)
    print(f"eval Medinan frac {y_eval.mean():.3f}, train {y_train.mean():.3f}")

    model = SentenceTransformer(MODEL, device="cpu")
    prefix = "query: " if "e5" in MODEL else ""
    def embed(texts):
        return model.encode([prefix + t for t in texts], batch_size=64,
                            convert_to_numpy=True, normalize_embeddings=True,
                            show_progress_bar=False)

    def pick(df, ks):
        d = df.set_index(["sura", "aya"])["translation"]
        return [str(d.get(k, "")) for k in ks]

    X_train = embed(pick(en, train_keys))
    clf = LogisticRegression(max_iter=3000).fit(X_train, y_train)
    print("classifier trained")

    rows = []
    for lang, f in sorted(langs.items()):
        df = read_corpus_csv(f)
        X = embed(pick(df, eval_keys))
        pred = clf.predict(X)
        rows.append({"language": lang,
                     "accuracy": round(float(accuracy_score(y_eval, pred)), 4),
                     "macro_f1": round(float(f1_score(y_eval, pred, average="macro")), 4)})
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(f"downstream_{TAG}.csv", index=False)

    joint = pd.read_csv("/mnt/user-data/uploads/Quran/joint_language_table.csv")
    m = out.merge(joint, on="language", how="inner")
    print(f"merged {len(m)} languages with joint table (missing: {sorted(set(out.language)-set(m.language))})")
    res = {}
    for axis in ["retrieval", "fertility", "joshi"]:
        rho, p = spearmanr(m.macro_f1, m[axis])
        res[axis] = {"rho": round(float(rho), 3), "p": float(p)}
        print(f"macro_f1 vs {axis}: rho={rho:.3f} p={p:.2e}")
    json.dump(res, open(f"downstream_{TAG}_correlations.json", "w"), indent=1)

if __name__ == "__main__":
    main()
