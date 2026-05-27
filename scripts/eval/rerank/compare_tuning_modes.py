# coding=utf-8
"""Compare independent vs unified (λ, β) tuning across backbones.

Strategy:
  1. Independent: tune (λ, β) per-backbone on its own val, apply to its test.
  2. Unified:     concat both val sets, tune (λ, β) once, apply to each test.

Objective for unified: minimize total weighted WER = (err_USR + err_AvHub) / (ref_USR + ref_AvHub)
"""
import json, math
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


def err_ref(recs, lam, beta):
    err = 0; ref = 0
    for r in recs:
        best = max(r['rows'], key=lambda x: x['sum_log'] + lam * x['log_p_len'] - beta * x['nfe'])
        err += best['ed']
        ref += r['n_gt_words']
    return err, ref


def tune_pooled(recs_list, with_nfe=True,
                lam_range=(0.0, 2.0, 0.05), beta_range=(0.0, 0.6, 0.025)):
    """Tune (λ, β) on pooled records, minimizing aggregate WER."""
    best = (1e9, 0.0, 0.0)
    l0, l1, ls = lam_range
    b0, b1, bs = beta_range
    li_n = int(round((l1 - l0) / ls))
    bi_n = int(round((b1 - b0) / bs)) if with_nfe else 0

    def agg_wer(lam, beta):
        E = 0; R = 0
        for recs in recs_list:
            e, r = err_ref(recs, lam, beta)
            E += e; R += r
        return 100.0 * E / max(1, R)

    for li in range(li_n + 1):
        lam = round(l0 + li * ls, 6)
        if with_nfe:
            for bi in range(bi_n + 1):
                beta = round(b0 + bi * bs, 6)
                w = agg_wer(lam, beta)
                if w < best[0]: best = (w, lam, beta)
        else:
            w = agg_wer(lam, 0.0)
            if w < best[0]: best = (w, lam, 0.0)
    # fine sweep
    _, lam0, beta0 = best
    best2 = best
    for li in range(-20, 21):
        lam = max(0.0, round(lam0 + li * 0.01, 4))
        if with_nfe:
            for bi in range(-15, 16):
                beta = max(0.0, round(beta0 + bi * 0.01, 4))
                w = agg_wer(lam, beta)
                if w < best2[0]: best2 = (w, lam, beta)
        else:
            w = agg_wer(lam, 0.0)
            if w < best2[0]: best2 = (w, lam, 0.0)
    return best2


def tune_single(recs, with_nfe=True,
                lam_range=(0.0, 2.0, 0.05), beta_range=(0.0, 0.6, 0.025)):
    return tune_pooled([recs], with_nfe=with_nfe, lam_range=lam_range, beta_range=beta_range)


def main():
    usr2_val   = Path('ckpt/usr2/dream_stage2/canvas32_b32_val')
    usr2_test  = Path('ckpt/usr2/dream_stage2/canvas32_b32_test')
    av_val     = Path('ckpt/avhubert/dream_stage2/canvas32_b32_val')
    av_test    = Path('ckpt/avhubert/dream_stage2/canvas32_b32_test')

    usr2_val_r = load_records(usr2_val)
    usr2_test_r = load_records(usr2_test)
    av_val_r   = load_records(av_val)
    av_test_r  = load_records(av_test)
    print(f'USR2  val: {len(usr2_val_r)} test: {len(usr2_test_r)}')
    print(f'AvHub val: {len(av_val_r)} test: {len(av_test_r)}')

    for tag, with_nfe in [('Row 3 (λ only)', False), ('Row 4 (λ + β)', True)]:
        print(f'\n========== {tag} ==========')

        # Independent
        _, lam_u, beta_u = tune_single(usr2_val_r, with_nfe=with_nfe)
        usr2_test_indep, _, _ = wer(usr2_test_r, lam_u, beta_u)
        _, lam_a, beta_a = tune_single(av_val_r, with_nfe=with_nfe)
        av_test_indep, _, _ = wer(av_test_r, lam_a, beta_a)

        print(f'Independent:')
        print(f'  USR2  (λ,β)=({lam_u:.3f},{beta_u:.3f})  test_WER={usr2_test_indep:.3f}%')
        print(f'  AvHub (λ,β)=({lam_a:.3f},{beta_a:.3f})  test_WER={av_test_indep:.3f}%')

        # Unified (pooled val)
        _, lam_p, beta_p = tune_pooled([usr2_val_r, av_val_r], with_nfe=with_nfe)
        usr2_test_uni, _, _ = wer(usr2_test_r, lam_p, beta_p)
        av_test_uni,   _, _ = wer(av_test_r,   lam_p, beta_p)
        print(f'Unified (pooled val):')
        print(f'  (λ,β)=({lam_p:.3f},{beta_p:.3f})')
        print(f'  USR2  test_WER={usr2_test_uni:.3f}%   (Δ vs indep: {usr2_test_uni - usr2_test_indep:+.3f})')
        print(f'  AvHub test_WER={av_test_uni:.3f}%   (Δ vs indep: {av_test_uni - av_test_indep:+.3f})')


if __name__ == '__main__':
    main()
