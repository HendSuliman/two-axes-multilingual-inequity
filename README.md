# Two Axes of Multilingual Inequity: Corpus, Code, and Results

Anonymous release accompanying the submission "Two Axes of Multilingual Inequity: Tokenizer
Fairness and Retrieval Coverage on a Meaning-Controlled 41-Language Corpus".

## What this is

A meaning-controlled multi-parallel corpus built from the Qur'an: 6,236 verses translated into
40 languages by professional translators within one publishing project (QuranEnc / IslamHouse),
verse-aligned with zero missing cells. There are 41 translation files, because Bosnian appears in
two translations; they are closely related (58% of verses identical), not independent. With the Arabic source text (Tanzil Uthmani) the corpus spans 41
languages in 42 files.

Three counts recur and are kept distinct: fertility statistics run over the 41 translation files;
analyses that correlate an axis with per-language covariates use one Bosnian translation
(`bosnian-rwwad`) and so have n = 40; the corpus as a resource is 41 languages.

On this corpus the paper measures (1) tokenizer fertility across eleven production tokenizers
(ten distinct vocabularies, since Aya-101 ships mT5's), (2) cross-lingual verse retrieval across
four multilingual sentence encoders, and (3) zero-shot transfer on two probes, a binary
Meccan/Medinan task and a 15-way sura task. It adds a FLORES-200 replication, a re-specified test
of script against resource tier, and five quantified robustness checks, including translator
variance across 33 translations of 14 languages and a test of translation style.

## Quick start

Python 3.10 or later.

```bash
pip install -r requirements.txt
python code/table3_partial_correlations.py        # Table 4, from shipped results   (~1 min, CPU)
python code/reanalysis_script_effect.py           # Section 6.1 and its tables      (~2 min, CPU)
python code/flores_fertility.py --summary-only    # FLORES table                    (~1 min, CPU)
python code/downstream_transfer.py --analysis-only
python code/robustness_checks.py duplicates length_probe
python code/tanzil_membership.py                  # Appendix D, training exposure (~1 min, CPU)
```

These six run offline from the shipped results and reproduce the paper's numbers exactly. Each
script runs from any working directory, unzips `data/translations.zip` into `corpus/` on first
use, and writes to `outputs/` so the shipped `results/` are never overwritten and can be diffed
against. Scripts that tokenize or embed text (the rest of `robustness_checks.py`,
`flores_fertility.py` without `--summary-only`, `flores_retrieval.py`, `downstream_transfer.py`
with an encoder name) download models from the Hugging Face Hub, and FLORES-200 from Meta's
official URL on first run.

## Layout

```
data/
  translations.zip              41 translation files (CSV: id, sura, aya, translation, footnotes)
  metadata/
    joint_language_table.csv    per-language script, Joshi (2020) resource tier (verified against
                                the released lang2tax taxonomy), GPT-4o fertility, and retrieval
                                pooled over the four encoders (use results/retrieval_per_language.csv
                                for per-encoder retrieval)
    sura_labels.csv             Meccan/Medinan label per sura (86/28, standard division)
    duplicate_groups_english.csv  91 near-duplicate verse groups (token Jaccard >= 0.8, English)

code/
  _paths.py                     repository paths shared by the scripts
  -- experiments (Colab notebooks; GPU recommended for 02 and 04)
  01_tokenizer_fertility.ipynb  Experiment 1: fertility, 11 tokenizers
  02_crosslingual_retrieval.ipynb  Experiment 2: retrieval, 4 encoders; verifies tokenization
                                against the reference sentencepiece library before scoring
  03_stats_and_paper.ipynb      bootstrap CIs, correlations, Table 2
  04_flores_replication_and_downstream.ipynb  FLORES-200 replication and the binary probe,
                                with the corrected Table 3 cell
  -- scripts
  table3_partial_correlations.py  Table 4: per-encoder transfer, own-encoder retrieval control
  reanalysis_script_effect.py   Section 6.1: script vs tier (Freedman-Lane), variance
                                decomposition, Brahmic/Latin contrast, script-length/fit grid
  downstream_transfer.py        binary Meccan/Medinan probe (script version)
  flores_fertility.py           FLORES-200 fertility replication
  flores_retrieval.py           FLORES-200 retrieval-floor check for low-resource languages
  robustness_checks.py          near-duplicates, truncation, Bosnian pair, length probe,
                                tokenizer-mirror verification
  get_arabic_source.py          downloads the Tanzil Uthmani Arabic source text
  -- analyses added in revision (Colab notebooks)
  probe2_sura_colab.ipynb       second probe: 15-way sura classification (GPU)
  translator_variance_colab.ipynb  33 translations of 14 languages, 5 scripts
  adequacy_insertions_colab.ipynb  translation style: bracketed insertions (CPU) and reference-free
                                adequacy against the Arabic source with CometKiwi (GPU, gated model)
  sentencepiece_check_colab.ipynb  mT5 / XLM-R token counts against the reference sentencepiece
                                library (see "Changes" below)
  flores_spm_check_colab.ipynb  the same check on FLORES-200 and the Arabic source
  verify_table3_colab.ipynb     independent check of the Table 4 correction (Table 3 in August) (numpy/scipy only)
  tatoeba_comparison_colab.ipynb  the corpus against Tatoeba on the 26 shared languages (GPU)
  tanzil_membership.py          Appendix D: does membership in Tanzil/OPUS explain retrieval beyond tier?
  tanzil_overlap_colab.ipynb    Appendix D: is each corpus translation the one Tanzil/OPUS distributes? (CPU)
  translator_independence_colab.ipynb  are the translator-study translations independent of each other? (CPU)
  bible_comparison_colab.ipynb  the corpus against the eBible corpus: coverage, tokenizer audit,
                                translator variance, retrieval floor (CPU; retrieval part on GPU)

results/
  fertility_production_v3_spm.csv    tokens/verse, 11 tokenizers x 42 files, + ratios
  fertility_tokenizer_ci_v3_spm.csv  Table 2: mean with bootstrap CI, spread, p90/p10 with CI
  fertility_controlled_production.csv  matched-size English-only vs multilingual 32k BPE
  cost_simulation.csv                dollar and context-window conversions (GPT-4o)
  retrieval_pairs.csv                1,640 ordered language pairs x 4 encoders (top-1, MRR)
  retrieval_metadata.json            library versions and tokenization check for that run
  meaning_vs_language.csv            nearest-neighbour probe: same verse vs same language
  retrieval_per_language.csv         per-language, per-encoder retrieval means
  retrieval_encoder_ci.csv, retrieval_language_ci.csv  bootstrap CIs
  downstream_all_encoders.csv        binary probe, 41 files x 4 encoders
  table3_partial_correlations.csv    Table 4 (corrected)
  flores_fertility_by_language.csv   FLORES-200 devtest fertility per tokenizer and language
  flores_vs_corpus_summary.csv       FLORES table (tokenizer rank agreement in flores_rank_agreement.json)
  flores_retrieval_floor.csv         FLORES-200 English-paired retrieval for floor languages
  truncation_rates.csv               % verses over each encoder's sequence limit
  length_probe.csv                   length-only baseline for the binary probe
  reanalysis/                        outputs of reanalysis_script_effect.py
  probe2/                            second probe: scores, agreement, length baseline
  translator_variance/               translator study: fertility, retrieval and embedding spread
  adequacy/                          translation-style and adequacy results
  sentencepiece_check/               token-count verification for mT5 / XLM-R
  tatoeba/                           Tatoeba comparison: accuracy per language, agreement, spread
  tanzil/                            training-exposure checks against Tanzil/OPUS
  bible/                             eBible comparison: coverage, fertility, translator spread, retrieval
  superseded/                        August versions of the corrected files, kept for audit
```

## Changes since the August submission

**Table 4 control corrected (Table 3 in August).** The August per-encoder table correlated every encoder's transfer
with the `retrieval` column of `joint_language_table.csv`, which is the mean over all four
encoders. It now controls for each encoder's own retrieval accuracy
(`results/retrieval_per_language.csv`). Three of four retrieval correlations rise (MPNet 0.78 to
0.91); partial correlations of transfer with fertility are LaBSE -0.02, BGE-M3 -0.14, MPNet +0.14,
multilingual-E5 -0.37 [-0.61, -0.08], at n = 40 where the smallest detectable |rho| is 0.32.
`downstream_transfer.py` and the `04` notebook are fixed accordingly;
`table3_partial_correlations.py` prints the pooled version alongside for comparison.

**mT5 / Aya-101 and XLM-R token counts corrected.** Recounting with Google's reference
`sentencepiece` library on each model's own model file showed the August counts too high for 21 of
41 files (Yoruba by about 19%, Malayalam 9-11%, Persian 4-7%, Assamese 2%, under 1.5% elsewhere).
transformers 4.46 and 5.16 both agree with sentencepiece; the affected languages are those whose
text Unicode NFKC normalization changes, which is consistent with the tokenizers' built-in
normalization not having been applied in the original run. The other eight tokenizers reproduce
exactly. Table 2 changes for these two rows only (mT5 p90/p10 stays 1.71; XLM-R 1.93 to 1.92), the
script effect on tokens per byte rises slightly (mT5 0.77 to 0.78, XLM-R 0.80 to 0.82), and the
FLORES rank agreement moves from 0.915 to 0.924. `results/superseded/` keeps the August files.
The FLORES-side counts were unaffected.

**Retrieval re-run.** BGE-M3, multilingual-E5 and MPNet use XLM-R's SentencePiece vocabulary, so the
August retrieval run inherited the same normalization fault. Experiment 2 was re-run with the
identical protocol and 1,000-verse sample (`retrieval_metadata.json` records the library versions
and a tokenization check against the reference library, which all three pass). LaBSE reproduces
exactly. The other three agree with August at per-language rank rho >= 0.99. Mean top-1 moves from
0.758 to 0.765 (BGE-M3), from 0.651 to 0.668 (multilingual-E5) and from 0.317 to 0.320 (MPNet), and
Yoruba moves most. Every retrieval-derived result in `results/` has been regenerated; the August
retrieval files are in `results/superseded/` with an `_august` suffix. Language-level correlations
now all use n = 40, as the paper states.

**Added:** the re-specified script-vs-tier analysis, the second probe, the translator study, the
translation-style and adequacy checks, and the Tatoeba and Bible comparisons listed under `code/`.
The Bible comparison downloads eBible (BibleNLP/ebible on GitHub) at run time; the Bible texts are
not redistributed here, and each carries its own licence.

## Tokenizers and encoders

Tokenizers: tiktoken `o200k_base` / `cl100k_base`; Hugging Face `bert-base-multilingual-cased`,
`xlm-roberta-base`, `bigscience/bloom-560m`, `facebook/nllb-200-distilled-600M`,
`google/mt5-small` (also the Aya-101 open-release vocabulary), `google/byt5-small`; Llama-3 and
Gemma-2 via ungated mirrors (`unsloth/llama-3-8b`, `unsloth/gemma-2-2b`), verified to give token
counts identical to two further independent mirrors each (`robustness_checks.py mirrors`).

Encoders: `sentence-transformers/LaBSE` (limit 256),
`sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (128),
`intfloat/multilingual-e5-base` (512, "query:" prefix), `BAAI/bge-m3` (8,192).

Seeds are fixed: the retrieval sample, probe splits and bootstrap draws are identical across
languages and encoders.

## Licensing

- **Translations** (`data/translations.zip`): distributed by IslamHouse.com under the MIT
  License; see `DATA_LICENSE.md`. Redistributed unmodified, with per-file translation identifiers
  and provenance headers intact.
- **Arabic source**: not bundled; `code/get_arabic_source.py` fetches the Tanzil Uthmani text,
  whose terms permit redistribution of the unmodified text with attribution (https://tanzil.net).
- **FLORES-200**: CC-BY-SA 4.0, downloaded from Meta at run time, not redistributed.
- **Code and derived result files**: MIT License (`LICENSE`).

## Known limitations

Quantified in the paper's robustness section and Appendix D: near-duplicate verses (about 1.7% of
a 1,000-verse sample has an in-sample near-duplicate); MPNet truncation (up to 20% of verses in
high-fertility languages; the untruncated encoders carry the conclusions); translator choice
(median within-language max/min ratio 1.07, or 1.12 on independent pairs, against between-language p90/p10 of 1.4-6.5); translation
style (bracketed insertions, a median 4.2% of a file's bytes, move p90/p10 by at most 11% and leave
the script-fit result intact); and the length component of the binary probe (macro-F1 0.601,
uncorrelated with retrieval). The corpus controls meaning relative to independently sampled text;
it does not establish semantic equivalence across languages.
