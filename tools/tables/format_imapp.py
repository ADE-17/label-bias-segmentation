import pandas as pd
import numpy as np
import os

def load_metrics(csv_path, exp_name):
    if not os.path.exists(csv_path): return None
    df = pd.read_csv(csv_path)
    df = df[df['experiment'] == exp_name]
    if len(df) == 0: return None
    row = df.iloc[0]
    
    true_iou = row['test_true_iou'] * 100
    true_iou_gap = row['test_true_iou_gap'] * 100
    
    excess = row['cl_ler_clean'] / 1e6
    deficit = row['cl_ler_biased'] / 1e6
    
    rr = row['cl_rr']
    sdr_clean = row['cl_sdr_clean'] * 100
    sdr_biased = row['cl_sdr_biased'] * 100
    sdr_gap = row['cl_sdr_gap'] * 100
    
    return f"${true_iou:.1f}/{true_iou_gap:.1f}$", f"${excess:.2f}/{deficit:.2f}$", f"${rr:.2f}/{sdr_gap:.1f}$"

def get_row_data(exp_name):
    res = load_metrics('/path/to/output/evaluations_imapp/all_experiments_summary.csv', exp_name)
    if not res: res = ("--", "--", "--")
    return res

methods = [
    ("ERM", "imapp_baseline_r0"),
    ("EO", "imapp_fairness_eo_r0"),
    ("DP", "imapp_fairness_dp_r0"),
    ("EO+DP", "imapp_fairness_both_r0"),
    ("Adversarial", "imapp_adversarial_r0"),
    ("MMD", "imapp_mmd_r0"),
    ("CORAL", "imapp_coral_r0"),
    ("GCE", "imapp_gce_r0"),
    ("Bootstrapping", "imapp_bootstrapping_r0"),
    ("Asym. mask", "imapp_asym_r0"),
    ("$g$-conditioned", "imapp_style_r0"),
    ("Combined", "imapp_asym_style_r0"),
    ("Auto-conditioned", "imapp_hybrid_moe_r0")
]

lines = []
for name, exp_name in methods:
    r = get_row_data(exp_name)
    if name == "Combined":
        r = (f"\\mathbf{{{r[0].replace('$', '')}}}", r[1], r[2])
    
    line = f"& {name} & {r[0]} & {r[1]} & {r[2]} \\\\"
    lines.append(line)

print("\n".join(lines))
