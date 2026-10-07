#!/usr/bin/env bash
# Runs the roadmap end to end. Each step is skipped when its output exists, so re-running after a
# crash resumes (training steps also resume mid-epoch from runs/<name>/ckpt.pt).
#
#   CONFIG=configs/smoke.yaml scripts/run_all.sh          # synthetic smoke test
#   CONFIG=configs/default.yaml scripts/run_all.sh        # real run
#   STEPS="ablation eval" scripts/run_all.sh              # only some steps
#
# Headline runs (baseline, stage1, eki) train on subset_ids.txt. Table V rows train on
# ablation_ids.txt so all six rows share one training set.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${CONFIG:-configs/default.yaml}
PY=${PY:-$HOME/miniconda3/envs/eki/bin/python}
PY_OCR=${PY_OCR:-$HOME/miniconda3/envs/eki-ocr/bin/python}
OCR_DEVICE=${OCR_DEVICE:-gpu}
STEPS=${STEPS:-"probe data expert ocr baseline mine stage1 eki infer ablation eval"}  # + "sweep" for Fig. 6
EXTRA=${EXTRA:-}                                     # e.g. EXTRA="train.save_every=100"
cfg_get() { $PY -c "from eki.config import load; c = load('$CONFIG', '$EXTRA'.split()); print($1)"; }
WORK=$(cfg_get "c.work")
SPLITS=$(cfg_get "' '.join(c.data.test_lmdbs)")

run() { echo -e "\n=== $*"; "$@"; }
py() { run $PY "$@" --config "$CONFIG" --set $EXTRA; }
want() { [[ " $STEPS " == *" $1 "* ]]; }
done_() { [[ -e "$WORK/$1" ]]; }

train() {  # train <name> <stage> [args...]
  local name=$1; shift
  done_ "runs/$name/DONE" || py scripts/train_mllm.py --name "$name" --stage "$@"
}
infer_all() {
  for s in $SPLITS; do py scripts/infer.py --run "$1" --split "$s"; done
}

want probe    && { done_ p0_report.json || py scripts/p0_probe.py; }
want data     && { done_ boxes.jsonl || py scripts/p1_data.py; }
want expert   && {
  done_ expert_backbone.pt || py scripts/p2_expert.py train
  done_ expert/val_report.json || py scripts/p2_expert.py eval
  py scripts/p2_expert.py cache
}
want ocr      && {
  done_ ocr.jsonl || run $PY_OCR scripts/p3_ocr.py --config "$CONFIG" --device "$OCR_DEVICE"
  done_ tf_pairs.jsonl || py scripts/p3_tf.py
}
# Baseline = M_pre: Stage 2 recipe, original ViT, no FGRA, 1 epoch (DECISIONS.md)
want baseline && train baseline 2 --set stage2.use_fgra=false stage2.epochs=1
want mine     && {
  done_ runs/baseline/preds_subset_ids.jsonl.done || {
    py scripts/infer.py --run baseline --ids subset_ids.txt && touch "$WORK/runs/baseline/preds_subset_ids.jsonl.done"; }
  done_ hard_ids.txt || py scripts/p4_mine.py mine --preds runs/baseline/preds_subset_ids.jsonl
}
want stage1   && train stage1 1
want eki      && train eki 2 --vit stage1
want infer    && { infer_all baseline; infer_all eki; }

if want ablation; then
  A="--ids ablation_ids.txt"
  train abl_tf_s1   1 $A --set stage1.use_if=false
  train abl_if_s1   1 $A --set stage1.use_tf=false
  train abl_tfif_s1 1 $A
  train baseline_a  2 $A --set stage2.use_fgra=false
  train tf          2 $A --vit abl_tf_s1   --set stage2.use_fgra=false
  train if          2 $A --vit abl_if_s1   --set stage2.use_fgra=false
  train tfif        2 $A --vit abl_tfif_s1 --set stage2.use_fgra=false
  train fgra_only   2 $A
  train eki_a       2 $A --vit abl_tfif_s1
  for r in baseline_a tf if tfif fgra_only eki_a; do infer_all $r; done
fi

# Fig. 6 layer sweep (opt-in: STEPS="sweep"); l=1 is eki_a. 'last' = number of LLM layers.
if want sweep; then
  NL=$(cfg_get "(lambda a: getattr(a, 'text_config', a).num_hidden_layers)(__import__('transformers').AutoConfig.from_pretrained(c.model.mllm))")
  for l in 0 4 14 $NL; do
    train fgra_l$l 2 --ids ablation_ids.txt --vit abl_tfif_s1 --set stage2.fgra_layer=$l
    py scripts/infer.py --run fgra_l$l --split "${SPLITS%% *}"     # Fig. 6 uses the main split only
  done
fi

want eval && {
  shopt -s nullglob; sweep=("$WORK"/runs/fgra_l*); shopt -u nullglob
  py scripts/evaluate.py --runs baseline eki baseline_a tf if tfif fgra_only eki_a "${sweep[@]##*/}"
  if (( ${#sweep[@]} )); then py scripts/figures.py sweep; fi
  py scripts/figures.py pca --runs original baseline eki
}
echo "done: results in $WORK/results.csv"
