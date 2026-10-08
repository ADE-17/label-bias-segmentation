# Towards Fairness under Label Bias in Image Segmentation

**Impact, Measurement and Mitigation** · NeurIPS 2026

Aditya Parikh, Stella Frank, Sneha Das, Aasa Feragen · Technical University of Denmark

[**Project page**](https://ade-17.github.io/label-bias-segmentation/) ·
[**arXiv:2605.06891**](https://arxiv.org/abs/2605.06891) ·
[**PDF**](https://arxiv.org/pdf/2605.06891)

Label bias arises when annotation style or quality depends on the demographic group in the image.
This repository contains the full pipeline from the paper: controlled bias injection, a Confident
Learning audit that works without clean labels (LER, DR, RR), latent-space separability analysis,
baseline mitigations, and the proposed Subgroup-Conditioned Decoder (SCD / Auto-SCD), together with
the exact splits, seeds and per-experiment configs.

## Repository layout

```
labelbias/                 Python package (run modules with `python -m labelbias.<...>` from the repo root)
  data/
    celebamask.py          CelebAMask-HQ dataset + biased-label wrapper (erosion / dilation / wave)
    phc.py                 PhC-U373 cells: tint shortcut, fine (clean) vs coarse (biased) annotations
    imapp.py               IMA++ dermoscopy: clean / biased mask pairs per skin-tone group
    factory.py             Single --dataset switch: defaults, splits, dataset construction
    splits.py              Reproducible stratified K-fold split creation
  metrics.py               IoU / Dice and losses (combined, boundary)
  debiasing.py             Adversarial (GRL), fairness (DP / EO) and MMD / CORAL modules
  audit/
    confident_learning.py  CL audit: LER, DR, RR, joint matrix per group (CelebAMask-HQ, PhC)
    confident_learning_imapp.py   Same for skin-tone groups on IMA++
  train/
    erm.py                 ERM baseline; GCE and bootstrapping robust losses via --loss_mode
    invariance.py          Adversarial / fairness / MMD / CORAL mitigation training
    scd.py                 Proposed: asymmetric boundary masking, SCD (FiLM), and their combination
    auto_scd.py            Proposed: Auto-SCD (reference group selected from warm-up loss)
    imapp_erm.py           ERM baseline for IMA++
  evaluation/
    evaluate.py            True / observed IoU per group + CL audit (celebamask, phc)
    evaluate_imapp.py      Same for IMA++
    feature_analysis*.py   Latent-space separability (probe accuracy, silhouette, Fisher, MMD)
    calibration_ablation.py  Temperature-scaling sanity check of the CL thresholds
tools/
  tables/                  Turn evaluation CSVs into the paper tables
  preprocessing/           IMA++ pairing and ISIC skin-tone (ITA) estimation
scripts/                   SLURM job scripts with the exact commands used for every experiment
configs/
  splits/                  cv_splits.json, phc_splits.json, imapp_splits.json (15 % test + 5 folds)
  experiments/<exp>/fold_<k>.json   As-run argument dumps of every experiment
  imapp_processed.csv, ita_stats_kmeans.csv   Dataset CSVs (paths are placeholders)
docs/                      Project page (GitHub Pages, served from this folder)
```

Paper method → code:

| Paper | Module / flag |
|---|---|
| ERM, GCE, bootstrapping | `labelbias.train.erm` (`--loss_mode`) |
| Fairness EO / DP, adversarial, MMD, CORAL | `labelbias.train.invariance` (`--method`, `--fairness_mode`, `--domain_method`) |
| Asymmetric boundary masking | `labelbias.train.scd --method asymmetric` |
| SCD (subgroup-conditioned FiLM decoder) | `labelbias.train.scd --method style_cond` |
| SCD + Asym | `labelbias.train.scd --method asym_style` |
| Auto-SCD | `labelbias.train.auto_scd` |
| CL audit (LER, DR, RR) | `labelbias.audit.confident_learning`, run through `labelbias.evaluation.evaluate` |
| Separability analysis | `labelbias.evaluation.feature_analysis` |

## Datasets

| Key | Dataset | Groups | Bias source |
|---|---|---|---|
| `celebamask` | CelebAMask-HQ | gender attribute | synthetic erosion / dilation / wave of the mask |
| `phc` | PhC-U373 | synthetic colour tint | fine (clean) vs coarse (biased) human annotators |
| `imapp` | IMA++ | estimated skin tone (ITA) | consensus (clean) vs looser annotator (biased) |

Datasets are not redistributed. Pass `--data_root` or edit `DATASET_DEFAULTS` in
`labelbias/data/factory.py`. Placeholder paths in the code are `/path/to/...`.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+, PyTorch 2.x, `segmentation-models-pytorch`.

## Run

All commands run from the repository root.

```bash
# ERM baseline, CelebAMask-HQ, 50 % erosion bias on the female group, fold 0
python -m labelbias.train.erm --dataset celebamask --bias_mode erosion --bias_ratio 0.5 \
    --erosion_radius 15 --fold 0 --epochs 15 --seed 42 --bias_seed 42 --exp_name celeba_erm_r0.5

# Proposed SCD + asymmetric masking
python -m labelbias.train.scd --dataset celebamask --method asym_style --bias_mode erosion --bias_ratio 0.5 --erosion_radius 15 --fold 0 --epochs 15 --exp_name celeba_scd_asym_r0.5

# Auto-SCD (no reference group given)
python -m labelbias.train.auto_scd --dataset celebamask --bias_ratio 0.5 --erosion_radius 15 --fold 0 --epochs 15 --exp_name celeba_auto_scd_r0.5

# Evaluate: true / observed IoU per group + CL audit
python -m labelbias.evaluation.evaluate --dataset celebamask --experiments_dir <out>/experiments --experiments celeba_erm_r0.5 celeba_scd_asym_r0.5

# Separability of encoder features
python -m labelbias.evaluation.feature_analysis --experiments_dir <out>/experiments --experiments celeba_erm_r0.5
```

`scripts/*.sh` contain the exact command lines (all methods, bias levels, folds) used for the paper.

## Reproducibility

* **Splits**: `configs/splits/*.json`, generated by `labelbias/data/splits.py` with `random_seed=42`,
  `test_ratio=0.15`, stratified by group.
* **Seeds**: every training run uses `--seed 42` (init, shuffling) and `--bias_seed 42` (which
  samples of the biased group are corrupted); batch scripts export `PYTHONHASHSEED=42`.
* **Configs**: `configs/experiments/<exp>/fold_<k>.json` are the argument dumps written at
  training time, including the resolved bias configuration.

## Project page

`docs/` is a static site (no build step). To publish it, enable GitHub Pages in the repository
settings with source *Deploy from a branch*, branch `main`, folder `/docs`. Author photos go in
`docs/assets/authors/` as `stella_frank.jpg`, `sneha_das.jpg`, `aasa_feragen.jpg` (square crops;
missing files fall back to initials).

## Citation

```bibtex
@inproceedings{parikh2026labelbias,
  title     = {Towards Fairness under Label Bias in Image Segmentation: Impact, Measurement and Mitigation},
  author    = {Parikh, Aditya and Frank, Stella and Das, Sneha and Feragen, Aasa},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  eprint    = {2605.06891},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url       = {https://arxiv.org/abs/2605.06891}
}
```
