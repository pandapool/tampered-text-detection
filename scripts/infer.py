"""Phase 7: plain-prompt greedy inference -> preds jsonl, plus parse-failure rate and latency.

  python scripts/infer.py --run eki --split TestingSet            # first data.test_slice images
  python scripts/infer.py --run eki --split TestingSet --full     # whole split
  python scripts/infer.py --run baseline --ids subset_ids.txt     # e.g. for hard mining
  python scripts/infer.py --run original --split TestingSet       # untrained Qwen2.5-VL

Writes <work_dir>/runs/<run>/preds_<split>.jsonl with {"id", "text", "boxes", "ok"} and a summary
json next to it. Resumes: ids already in the output file are skipped.
"""
import json
import time

import torch

from eki import boxes as B
from eki import mllm
from eki.config import cli
from eki.data import Splits, read_ids, read_jsonl
from eki.samples import messages


def args_fn(p):
    p.add_argument("--run", required=True, help="run name under runs/, or 'original'")
    p.add_argument("--split", default=None)
    p.add_argument("--ids", default=None, help="id file under work_dir (instead of --split)")
    p.add_argument("--full", action="store_true")
    p.add_argument("--out", default=None)


@torch.no_grad()
def main():
    cfg, a = cli(__doc__, args_fn)
    run_dir = cfg.work / "runs" / a.run
    run_dir.mkdir(parents=True, exist_ok=True)
    S = Splits(cfg)
    if a.ids:
        ids, tag = read_ids(cfg.work / a.ids), a.ids.rsplit(".", 1)[0]
    else:
        n = len(S.db(a.split))
        ids, tag = [f"{a.split}:{i}" for i in range(n if a.full else min(n, cfg.data.test_slice))], a.split
    out = cfg.work / a.out if a.out else run_dir / f"preds_{tag}.jsonl"
    done = {r["id"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [i for i in ids if i not in done]
    print(f"{len(todo)} / {len(ids)} images to run -> {out}")

    proc = mllm.processor(cfg)
    if a.run == "original":
        model = mllm.load_base(cfg).eval()
        model.config.use_cache = True
    else:
        model = mllm.load_for_inference(cfg, run_dir)
    prompt = proc.apply_chat_template(messages(cfg.prompts["global"]), tokenize=False, add_generation_prompt=True)

    times = []
    with open(out, "a") as f:
        for k, sid in enumerate(todo):
            enc = proc(text=[prompt], images=[S.image(sid)], return_tensors="pt").to("cuda")
            torch.cuda.synchronize(); t0 = time.perf_counter()
            gen = model.generate(**enc, max_new_tokens=cfg.infer.max_new_tokens, do_sample=False,
                                 use_cache=True)
            torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
            text = proc.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            bx, ok = B.parse_answer(text, cfg.data.img_size)
            f.write(json.dumps({"id": sid, "text": text, "boxes": bx, "ok": ok}) + "\n")
            if k % 50 == 0:
                f.flush()
                print(f"[{k}/{len(todo)}] {sid}: {text[:100]!r}", flush=True)

    rows = read_jsonl(out)
    summ = {"n": len(rows), "parse_fail_rate": sum(not r["ok"] for r in rows) / max(1, len(rows)),
            "ms_per_image": 1000 * sum(times[1:]) / max(1, len(times) - 1) if len(times) > 1 else None,
            "pred_real_rate": sum(not r["boxes"] for r in rows) / max(1, len(rows))}
    out.with_suffix(".summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))
    if summ["parse_fail_rate"] >= 0.01:
        print("GATE FAIL: parse failure rate >= 1%")


if __name__ == "__main__":
    main()
