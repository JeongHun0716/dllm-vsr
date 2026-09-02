# coding=utf-8
"""Single-setting Dream (dLLM) VSR evaluation, sharded across GPUs.

Runs ONE decoding configuration over the whole test set, split into disjoint
shards across processes, and reports a single exact WER. Use this for the
Implicit-length / oracle-length decoding (single fixed or oracle canvas, no reranking).

Cache strategy (speed/accuracy trade-off; cache_mode=none is the exact reference):
  eval.cache_mode ∈ {none, prefix, dual}

Launch (8-way shard, one command, one exact WER):
  accelerate launch --config_file accelerate_configs/8_gpu.yaml \\
    eval/eval_main.py config=configs/lrs3/usr2_v2.yaml \\
    experiment.eval_checkpoint=ckpt/usr2/dream_stage2 \\
    eval.alg=threshold eval.block_size=32 eval.threshold=0.9 eval.cache_mode=none
"""
import os
import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from training.utils import get_config
from eval.eval_core import (
    build_model_for_eval, build_sharded_loader, resolve_tail_token_id, run_one_setting,
)
from evaluation.fast_dream_generate import GENERATORS

logger = get_logger(__name__, log_level="INFO")


def resolve_cache_mode(config):
    """eval.cache_mode ∈ {none, prefix, dual}."""
    cache_mode = str(config.eval.get("cache_mode", "none"))
    if cache_mode not in GENERATORS:
        raise ValueError(f"eval.cache_mode must be one of {list(GENERATORS)}, got '{cache_mode}'")
    return cache_mode


def build_out_dir(config, *, alg, block_size, steps, cache_mode, threshold, fixed_k,
                  factor, answer_len, tail_token_id, model, tokenizer, ckpt_dir,
                  use_oracle_length=False):
    cache_tag = {"none": "nc", "prefix": "pc", "dual": "dc"}[cache_mode]
    mode_tag = (f"th{str(threshold).replace('.', 'p')}" if alg == "threshold"
                else f"k{fixed_k}" if alg == "fixed_k"
                else f"f{str(factor).replace('.', 'p')}")
    # Oracle length ignores answer_len (ground-truth length per sample), so tag it
    # separately — otherwise the fixed-canvas and oracle runs collide on one dir.
    slot_tag = "loracle" if use_oracle_length else f"l{answer_len}"
    if tail_token_id == int(model.pad_id):
        tail_tag = "tailpad"
    elif tail_token_id == int(tokenizer.eos_token_id):
        tail_tag = "taileos"
    else:
        tail_tag = f"tail{tail_token_id}"
    try:
        ckpt_step = int(str(ckpt_dir).rstrip("/").split("checkpoint-")[-1].split("/")[0])
        suffix = f"_ckpt{ckpt_step}"
    except (ValueError, IndexError):
        suffix = ""
    return Path(config.experiment.output_dir) / (
        f"eval_dream_{alg}_b{block_size}_s{steps}_{mode_tag}_{cache_tag}_{tail_tag}_{slot_tag}{suffix}"
    )


def main():
    logging.basicConfig(level=logging.INFO)
    config = get_config()

    accelerator = Accelerator(mixed_precision=config.training.mixed_precision)
    if config.training.seed is not None:
        set_seed(config.training.seed)

    model, tokenizer, mask_id, dllm_cfg_yaml, ckpt_dir = build_model_for_eval(config, accelerator)
    loader, n_total_set, n_shard = build_sharded_loader(
        config, accelerator, tokenizer_path=dllm_cfg_yaml.tokenizer_path,
        num_workers=int(config.dataset.dataloader.get("num_workers", 0)),
        split=str(config.eval.get("split", "test")),
    )

    cache_mode = resolve_cache_mode(config)
    alg = str(config.eval.get("alg", "threshold"))
    threshold = float(config.eval.get("threshold", 0.9))
    fixed_k = int(config.eval.get("fixed_k", 1))
    factor = float(config.eval.get("factor", 1.0))
    block_size = int(config.eval.get("block_size", 32))
    steps = int(config.eval.get("steps", 128))
    answer_len = int(config.eval.get("answer_len", 32))
    max_samples = int(config.eval.get("max_samples", -1))
    tail_token_id = resolve_tail_token_id(config, model, tokenizer)
    use_oracle = bool(model.config.use_oracle_length)

    logger.info(
        f"[EVAL-MAIN] world={accelerator.num_processes} test_set={n_total_set} this_shard={n_shard} "
        f"cache_mode={cache_mode} alg={alg} block_size={block_size} steps={steps} "
        f"threshold={threshold} fixed_k={fixed_k} factor={factor} answer_len={answer_len} "
        f"tail_token_id={tail_token_id} oracle={use_oracle}"
    )

    metrics, samples = run_one_setting(
        model=model, loader=loader, tokenizer=tokenizer, accelerator=accelerator,
        mask_id=mask_id, cache_mode=cache_mode, alg=alg, threshold=threshold,
        fixed_k=fixed_k, factor=factor, block_size=block_size, steps=steps,
        answer_len=answer_len, tail_token_id=tail_token_id, use_oracle_length=use_oracle,
        max_samples=max_samples, return_samples=True, tag="main",
    )

    if accelerator.is_main_process:
        rec = {
            "alg": alg, "block_size": block_size, "steps": steps, "cache_mode": cache_mode,
            "threshold": threshold, "fixed_k": fixed_k, "factor": factor,
            "answer_len": answer_len, "tail_token_id": tail_token_id, "ckpt": str(ckpt_dir),
            **metrics,
        }
        out_dir = build_out_dir(
            config, alg=alg, block_size=block_size, steps=steps, cache_mode=cache_mode,
            threshold=threshold, fixed_k=fixed_k, factor=factor, answer_len=answer_len,
            tail_token_id=tail_token_id, model=model, tokenizer=tokenizer, ckpt_dir=ckpt_dir,
            use_oracle_length=use_oracle,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "result.json", "w") as f:
            json.dump(rec, f, indent=2)
        with open(out_dir / "samples.jsonl", "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        logger.info(f"[EVAL-MAIN][RESULT] alg={alg} b={block_size} cache={cache_mode} "
                    f"WER={metrics['wer']:.2f}%  avg_nfe={metrics['avg_nfe']:.1f}  "
                    f"({metrics['n_samples']} samples)  → {out_dir/'result.json'}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
