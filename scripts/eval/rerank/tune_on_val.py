# coding=utf-8
"""Tune (λ, β) on val multi-K probe, apply to test multi-K probe.

Row 3 (+λ rerank): tune λ only on val with β=0
Row 4 (+λ+β):      tune (λ, β) jointly on val
Row 1 (sum_log):   λ=0, β=0 baseline
Row 2 (lenpred):   pick K=pred_K (argmax of lenpred)

Usage:
  python scripts/eval/rerank/tune_on_val.py \
      --val-dir <val_probe_dir> --test-dir <test_probe_dir>
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


def wer(recs, lam, beta):
    err = 0; ref = 0
    for r in recs:
        best = max(r['rows'], key=lambda x: x['sum_log'] + lam * x['log_p_len'] - beta * x['nfe'])
        err += best['ed']
        ref += r['n_gt_words']
    return 100.0 * err / max(1, ref), err, ref


def wer_baseline_sumlog(recs):
    """Row 1: argmax sum_log_conf only (no λ, no β)."""
    return wer(recs, 0.0, 0.0)


def wer_lenpred_only(recs):
    """Row 2: pick K = pred_K (argmax of lenpred), no rerank."""
    err = 0; ref = 0
    for r in recs:
        # max log_p_len = argmax_K of p(K|x)
        best = max(r['rows'], key=lambda x: x['log_p_len'])
        err += best['ed']
        ref += r['n_gt_words']
    return 100.0 * err / max(1, ref), err, ref


def tune(recs, with_nfe=True,
         lam_range=(0.0, 2.0, 0.05), beta_range=(0.0, 0.6, 0.025)):
    best = (1e9, 0.0, 0.0)
    l0, l1, ls = lam_range
    b0, b1, bs = beta_range
    li_n = int(round((l1 - l0) / ls))
    bi_n = int(round((b1 - b0) / bs)) if with_nfe else 0
    for li in range(li_n + 1):
        lam = round(l0 + li * ls, 6)
        if with_nfe:
            for bi in range(bi_n + 1):
                beta = round(b0 + bi * bs, 6)
                w, _, _ = wer(recs, lam, beta)
                if w < best[0]: best = (w, lam, beta)
        else:
            w, _, _ = wer(recs, lam, 0.0)
            if w < best[0]: best = (w, lam, 0.0)
    # fine sweep around coarse winner
    _, lam0, beta0 = best
    best2 = best
    for li in range(-20, 21):
        lam = max(0.0, round(lam0 + li * 0.01, 4))
        if with_nfe:
            for bi in range(-15, 16):
                beta = max(0.0, round(beta0 + bi * 0.01, 4))
                w, _, _ = wer(recs, lam, beta)
                if w < best2[0]: best2 = (w, lam, beta)
        else:
            w, _, _ = wer(recs, lam, 0.0)
            if w < best2[0]: best2 = (w, lam, 0.0)
    return best2  # (val_wer, lam, beta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val-dir',  required=True)
    ap.add_argument('--test-dir', required=True)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    print(f'=== {args.tag} ===')
    print(f'val:  {args.val_dir}')
    print(f'test: {args.test_dir}')

    val_recs  = load_records(args.val_dir)
    test_recs = load_records(args.test_dir)
    val_ngt  = sum(r['n_gt_words'] for r in val_recs)
    test_ngt = sum(r['n_gt_words'] for r in test_recs)
    print(f'val records: {len(val_recs)}  total_words: {val_ngt}')
    print(f'test records: {len(test_recs)}  total_words: {test_ngt}')

    # Row 1: sum_log baseline (no rerank)
    val_b1, _, _ = wer_baseline_sumlog(val_recs)
    test_b1, _, _ = wer_baseline_sumlog(test_recs)
    print(f'\n[Row 1 sum_log_conf, λ=0 β=0]')
    print(f'  val  WER = {val_b1:.3f}%')
    print(f'  test WER = {test_b1:.3f}%')

    # Row 2: pick K = pred_K (lenpred top1)
    val_b2, _, _ = wer_lenpred_only(val_recs)
    test_b2, _, _ = wer_lenpred_only(test_recs)
    print(f'\n[Row 2 lenpred top1 only]')
    print(f'  val  WER = {val_b2:.3f}%')
    print(f'  test WER = {test_b2:.3f}%')

    # Row 3: tune λ only on val
    val_w3, lam3, _ = tune(val_recs, with_nfe=False)
    test_w3, _, _ = wer(test_recs, lam3, 0.0)
    print(f'\n[Row 3 +λ rerank (tuned on val, β=0)]')
    print(f'  val-tuned λ = {lam3:.3f}')
    print(f'  val  WER (at this λ) = {val_w3:.3f}%')
    print(f'  test WER (apply λ)    = {test_w3:.3f}%')

    # Row 4: tune (λ, β) on val
    val_w4, lam4, beta4 = tune(val_recs, with_nfe=True)
    test_w4, _, _ = wer(test_recs, lam4, beta4)
    print(f'\n[Row 4 Full +λ+β (tuned on val)]')
    print(f'  val-tuned (λ, β) = ({lam4:.3f}, {beta4:.3f})')
    print(f'  val  WER (at this λ,β) = {val_w4:.3f}%')
    print(f'  test WER (apply λ,β)    = {test_w4:.3f}%')

    # Also report biased single-tune on test (= what we had before, leakage)
    test_w3_leak, lam3_leak, _ = tune(test_recs, with_nfe=False)
    test_w4_leak, lam4_leak, beta4_leak = tune(test_recs, with_nfe=True)
    print(f'\n[BIASED — single-tune on test for reference]')
    print(f'  Row 3 leak: λ={lam3_leak:.3f}  test WER = {test_w3_leak:.3f}%')
    print(f'  Row 4 leak: (λ,β)=({lam4_leak:.3f},{beta4_leak:.3f})  test WER = {test_w4_leak:.3f}%')

    out = {
        'tag': args.tag,
        'val_dir':  args.val_dir,
        'test_dir': args.test_dir,
        'val_n':  len(val_recs),  'val_ngt':  val_ngt,
        'test_n': len(test_recs), 'test_ngt': test_ngt,
        'row1_sumlog':   {'val': val_b1, 'test': test_b1},
        'row2_lenpred1': {'val': val_b2, 'test': test_b2},
        'row3_lambda':   {'lam_val': lam3, 'val_wer': val_w3, 'test_wer': test_w3,
                          'leak_lam': lam3_leak, 'leak_test_wer': test_w3_leak},
        'row4_full':     {'lam_val': lam4, 'beta_val': beta4, 'val_wer': val_w4, 'test_wer': test_w4,
                          'leak_lam': lam4_leak, 'leak_beta': beta4_leak, 'leak_test_wer': test_w4_leak},
    }
    return out


if __name__ == '__main__':
    main()
