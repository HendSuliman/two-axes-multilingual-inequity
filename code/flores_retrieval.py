#!/usr/bin/env python3
"""Retrieval-floor replication on FLORES-200 devtest: same four encoders,
English-paired retrieval for the corpus's floor languages plus references."""
import numpy as np, pandas as pd
from pathlib import Path

LANGS = {  # corpus name -> flores code
    "moore":"mos_Latn","oromo":"gaz_Latn","asante":"twi_Latn","lingala":"lin_Latn",
    "yoruba":"yor_Latn","kurdish":"ckb_Arab","khmer":"khm_Khmr","uyghur":"uig_Arab",
    "german":"deu_Latn","tamil":"tam_Taml",
}
ENC = {"labse":"sentence-transformers/LaBSE",
       "mpnet":"sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
       "e5":"intfloat/multilingual-e5-base",
       "bgem3":"BAAI/bge-m3"}
FL = Path("flores200_dataset/devtest")
RNG = np.random.default_rng(42)

def main():
    from sentence_transformers import SentenceTransformer
    sents = {c: (FL/f"{c}.devtest").read_text(encoding="utf-8").rstrip("\n").split("\n")
             for c in list(LANGS.values())+["eng_Latn"]}
    rows=[]
    for tag,name in ENC.items():
        model = SentenceTransformer(name, device="cpu")
        prefix = "query: " if "e5" in name else ""
        emb = {}
        for c,txts in sents.items():
            emb[c] = model.encode([prefix+t for t in txts], batch_size=64,
                                  convert_to_numpy=True, normalize_embeddings=True)
            print(f"{tag}: embedded {c}", flush=True)
        E = emb["eng_Latn"]
        for lang,c in LANGS.items():
            X = emb[c]
            s1 = (E @ X.T).argmax(1) == np.arange(len(E))   # eng -> X
            s2 = (X @ E.T).argmax(1) == np.arange(len(X))   # X -> eng
            both = np.concatenate([s1,s2])
            boots=[np.mean(RNG.choice(both,len(both),replace=True)) for _ in range(1000)]
            rows.append({"encoder":tag,"language":lang,"flores_top1":float(both.mean()),
                         "ci_lo":float(np.percentile(boots,2.5)),"ci_hi":float(np.percentile(boots,97.5))})
            print(rows[-1], flush=True)
        del model, emb
    pd.DataFrame(rows).to_csv("flores_retrieval_floor.csv", index=False)
    print("DONE")

if __name__ == "__main__":
    main()
