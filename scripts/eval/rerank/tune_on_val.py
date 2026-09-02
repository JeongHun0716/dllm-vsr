# coding=utf-8
"""Tune (λ, β) on the val multi-K decode output, apply to the test decode output.

Decoding dumps the full length-candidate pool (K_pred ± 5); reranking considers
candidates within radius R of the length predictor's top-1 (R=3 default, the
paper's operating point). Tuning follows the paper protocol: grid search over
λ, β ∈ [0, 1] with step 0.1 on the val set, then apply the winner to test.

Reported settings (rerank ablation rows in the paper):
  - sequence confidence only: λ=0, β=0
  - length-predictor top-1: pick K = pred_K (no reranking)
  - + length-prediction score (λ): tune λ on val with β=0
  - + iteration penalty (β), paper setting: tune (λ, β) jointly on val

Usage:
  python scripts/eval/rerank/tune_on_val.py \
      --val-dir <val_decode_dir> --test-dir <test_decode_dir> [--R 3]
"""
import json, math, argparse
from pathlib import Path


def load_records(root):
    records = []
    for f in sorted(Path(root).glob('shard_*_raw.jsonl')):
        for line in open(f):
            try: r = json.loads(line)
            except: continue
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


def rows_within(r, R):
    """Rows within radius R of pred_K (all rows if R is None/negative)."""
    if R is None or R < 0:
        return r['rows']
    pk = pred_K(r)
    rows = [x for x in r['rows'] if abs(x['K'] - pk) <= R]
    return rows if rows else r['rows']


def wer(recs, lam, beta, R=None):
    err = 0; ref = 0
    for r in recs:
        rows = rows_within(r, R)
        best = max(rows, key=lambda x: x['sum_log'] + lam * x['log_p_len'] - beta * x['nfe'])
        err += best['ed']
        ref += r['n_gt_words']
    return 100.0 * err / max(1, ref), err, ref


def wer_baseline_sumlog(recs, R=None):
    """argmax sum_log_conf only (no λ, no β)."""
    return wer(recs, 0.0, 0.0, R)


def wer_lenpred_only(recs):
    """pick K = pred_K (argmax of lenpred), no rerank."""
    err = 0; ref = 0
    for r in recs:
        # max log_p_len = argmax_K of p(K|x)
        best = max(r['rows'], key=lambda x: x['log_p_len'])
        err += best['ed']
        ref += r['n_gt_words']
    return 100.0 * err / max(1, ref), err, ref


def tune(recs, with_nfe=True, R=None):
    """Paper grid: λ, β ∈ [0, 1], step 0.1."""
    best = (1e9, 0.0, 0.0)
    for li in range(11):
        lam = round(0.1 * li, 1)
        betas = [round(0.1 * bi, 1) for bi in range(11)] if with_nfe else [0.0]
        for beta in betas:
            w, _, _ = wer(recs, lam, beta, R)
            if w < best[0]: best = (w, lam, beta)
    return best  # (val_wer, lam, beta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val-dir',  required=True)
    ap.add_argument('--test-dir', required=True)
    ap.add_argument('--R', type=int, default=3,
                    help='rerank search radius around pred_K (negative = no filter)')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    R = args.R

    print(f'=== {args.tag} ===')
    print(f'val:  {args.val_dir}')
    print(f'test: {args.test_dir}')
    print(f'rerank search radius R = {R}')

    val_recs  = load_records(args.val_dir)
    test_recs = load_records(args.test_dir)
    val_ngt  = sum(r['n_gt_words'] for r in val_recs)
    test_ngt = sum(r['n_gt_words'] for r in test_recs)
    print(f'val records: {len(val_recs)}  total_words: {val_ngt}')
    print(f'test records: {len(test_recs)}  total_words: {test_ngt}')

    # sum_log baseline (no rerank)
    val_b1, _, _ = wer_baseline_sumlog(val_recs, R)
    test_b1, _, _ = wer_baseline_sumlog(test_recs, R)
    print(f'\n[sequence confidence only, λ=0 β=0]')
    print(f'  val  WER = {val_b1:.3f}%')
    print(f'  test WER = {test_b1:.3f}%')

    # pick K = pred_K (lenpred top1)
    val_b2, _, _ = wer_lenpred_only(val_recs)
    test_b2, _, _ = wer_lenpred_only(test_recs)
    print(f'\n[length-predictor top-1, no reranking]')
    print(f'  val  WER = {val_b2:.3f}%')
    print(f'  test WER = {test_b2:.3f}%')

    # tune λ only on val
    val_w3, lam3, _ = tune(val_recs, with_nfe=False, R=R)
    test_w3, _, _ = wer(test_recs, lam3, 0.0, R)
    print(f'\n[+ length-prediction score (λ tuned on val, β=0)]')
    print(f'  val-tuned λ = {lam3:.1f}')
    print(f'  val  WER (at this λ) = {val_w3:.3f}%')
    print(f'  test WER (apply λ)    = {test_w3:.3f}%')

    # tune (λ, β) on val — paper setting
    val_w4, lam4, beta4 = tune(val_recs, with_nfe=True, R=R)
    test_w4, _, _ = wer(test_recs, lam4, beta4, R)
    print(f'\n[+ iteration penalty ((λ, β) tuned on val) — paper setting]')
    print(f'  val-tuned (λ, β) = ({lam4:.1f}, {beta4:.1f})')
    print(f'  val  WER (at this λ,β) = {val_w4:.3f}%')
    print(f'  test WER (apply λ,β)    = {test_w4:.3f}%')

    out = {
        'tag': args.tag,
        'val_dir':  args.val_dir,
        'test_dir': args.test_dir,
        'R': R,
        'val_n':  len(val_recs),  'val_ngt':  val_ngt,
        'test_n': len(test_recs), 'test_ngt': test_ngt,
        'sumlog_baseline': {'val': val_b1, 'test': test_b1},
        'lenpred_top1':    {'val': val_b2, 'test': test_b2},
        'lambda_only':     {'lam_val': lam3, 'val_wer': val_w3, 'test_wer': test_w3},
        'lambda_beta':     {'lam_val': lam4, 'beta_val': beta4, 'val_wer': val_w4, 'test_wer': test_w4},
    }
    return out


if __name__ == '__main__':
    main()
