# coding=utf-8
"""Apply a fixed (lambda, beta) rerank to multi-K canvas decode outputs.

Decoding dumps the full length-candidate pool (K_pred ± 5) once per utterance;
this script performs the radius-R rerank on top of it: candidates with
|K - pred_K| <= R are kept (R=3 default, the paper's operating point), the
fixed (lambda, beta) rerank picks the transcript, and the WER is reported.
It also prints the full R=1..5 sweep at the same (lambda, beta).

Default (lambda, beta) match the paper's val-tuned values for LRS3:
  - USR2:     (0.9, 0.6)
  - AvHubert: (0.9, 0.7)

Skips val tuning — assumes the paper's best hyperparameters and reports WER on test.
For the full val-tune-then-apply pipeline, use scripts/eval/rerank/tune_on_val.py.

Usage:
  python scripts/eval/rerank/apply_fixed_weights.py \
      --test-dir ckpt/usr2/dream_stage2/canvas32_b32_test \
      --lam 0.9 --beta 0.6 --R 3 --tag usr2
"""
import argparse, math, json
from pathlib import Path


def load_records(root):
    records = []
    for f in sorted(Path(root).glob('shard_*_raw.jsonl')):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get('n_gt_words', 0) == 0:
                continue
            for row in r['rows']:
                cc = row.get('commit_confs', [])
                row['sum_log'] = sum(math.log(max(c, 1e-12)) for c in cc) if cc else float('-inf')
            records.append(r)
    return records


def pred_K(r):
    """Length-predictor center K for a record (fallback: argmax p_len row)."""
    if 'pred_K_lenpred' in r:
        return r['pred_K_lenpred']
    return max(r['rows'], key=lambda x: x.get('log_p_len', float('-inf')))['K']


def wer(recs, lam, beta, R=None):
    err = 0
    ref = 0
    for r in recs:
        rows = r['rows']
        if R is not None:
            pk = pred_K(r)
            filt = [x for x in rows if abs(x['K'] - pk) <= R]
            if filt:
                rows = filt
        best = max(rows, key=lambda x: x['sum_log'] + lam * x['log_p_len'] - beta * x['nfe'])
        err += best['ed']
        ref += r['n_gt_words']
    return 100.0 * err / max(1, ref), err, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-dir', required=True, help='multi-K canvas decode output dir (shard_*_raw.jsonl)')
    ap.add_argument('--lam', type=float, required=True, help='length-prior weight (paper: 0.9)')
    ap.add_argument('--beta', type=float, required=True, help='nfe weight (paper: 0.6 USR2, 0.7 AvHub)')
    ap.add_argument('--R', type=int, default=3, help='search radius: keep candidates with |K - pred_K| <= R (default 3)')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    recs = load_records(args.test_dir)
    ref_total = sum(r['n_gt_words'] for r in recs)
    print(f"=== {args.tag} ===")
    print(f"  test-dir: {args.test_dir}")
    print(f"  records:  {len(recs)} utterances, {ref_total} GT words")
    print(f"  (lam, beta) = ({args.lam:.2f}, {args.beta:.2f}), R = {args.R}")

    w, err, ref = wer(recs, args.lam, args.beta, R=args.R)
    print(f"  test WER = {w:.3f}%   (err={err}, ref={ref})")

    print(f"=== search-radius sweep (lam={args.lam}, beta={args.beta}) ===")
    for R in range(1, 6):
        w, err, ref = wer(recs, args.lam, args.beta, R=R)
        mark = '>' if R == args.R else ' '
        print(f"{mark} R={R} ({2 * R + 1} cand): WER = {w:.3f}%")


if __name__ == '__main__':
    main()
