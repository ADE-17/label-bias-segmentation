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
    
    excess = row['cl_female_excess'] / 1e6
    deficit = row['cl_female_deficit'] / 1e6
    
    rr = row['cl_rr']
    sdr = row['cl_sdr_female'] * 100
    
    return f"${true_iou:.1f}/{true_iou_gap:.1f}$", f"${excess:.1f}/{deficit:.1f}$", f"${rr:.2f}/{sdr:.1f}$"

def get_row_data(exp_base):
    res_50 = load_metrics('/path/to/output/evaluations_celeba_mitig_50/all_experiments_summary.csv', f'celeba_{exp_base}_r0.50')
    if not res_50: res_50 = load_metrics('/path/to/output/evaluations_celeba_mitig_50/all_experiments_summary.csv', f'{exp_base}_baseline_r50_e15')
    if not res_50: res_50 = ("--", "--", "--")
    
    res_100 = load_metrics('/path/to/output/evaluations_celeba_mitig_100/all_experiments_summary.csv', f'celeba_{exp_base}_r1.0')
    if not res_100: res_100 = ("--", "--", "--")
    
    res_phc = load_metrics('/path/to/output/evaluations_phc_mitig_100/all_experiments_summary.csv', f'phc_{exp_base}_r1.0')
    if res_phc:
        df = pd.read_csv('/path/to/output/evaluations_phc_mitig_100/all_experiments_summary.csv')
        df = df[df['experiment'] == f'phc_{exp_base}_r1.0']
        row = df.iloc[0]
        t_iou, t_iou_gap = row['test_true_iou']*100, row['test_true_iou_gap']*100
        exc, defc = row['cl_female_excess']/1e6, row['cl_female_deficit']/1e6
        rr, sdr = row['cl_rr'], row['cl_sdr_female']*100
        res_phc = (f"${t_iou:.1f}/{t_iou_gap:.1f}$", f"${exc:.2f}/{defc:.2f}$", f"${rr:.2f}/{sdr:.1f}$")
    else:
        res_phc = ("--", "--", "--")
        
    # IMA++ (r=0.0)
    exp_name_imapp = f'imapp_{exp_base}_r0'
    if exp_base == "asym": exp_name_imapp = "imapp_asym_r0"
    elif exp_base == "style": exp_name_imapp = "imapp_style_r0"
    elif exp_base == "asym_style": exp_name_imapp = "imapp_asym_style_r0"
    elif exp_base == "hybrid_moe": exp_name_imapp = "imapp_hybrid_moe_r0"
        
    res_imapp = ("--", "--", "--")
    csv_imapp = '/path/to/output/evaluations_imapp/all_experiments_summary.csv'
    if os.path.exists(csv_imapp):
        df_i = pd.read_csv(csv_imapp)
        df_i = df_i[df_i['experiment'] == exp_name_imapp]
        if len(df_i) > 0:
            row = df_i.iloc[0]
            t_iou, t_iou_gap = row['test_true_iou']*100, row['test_true_iou_gap']*100
            exc, defc = row['cl_ler_clean']/1e6, row['cl_ler_biased']/1e6
            rr, sdr_gap = row['cl_rr'], row['cl_sdr_gap']*100
            rr_str = f"{rr:.2f}" if rr < 1000 else "\\infty"
            res_imapp = (f"${t_iou:.1f}/{t_iou_gap:.1f}$", f"${exc:.2f}/{defc:.2f}$", f"${rr_str}/{sdr_gap:.1f}$")

    # IMA++ (r=0.5)
    exp_name_imapp_50 = f'imapp_{exp_base}_r50'
    if exp_base == "asym": exp_name_imapp_50 = "imapp_asym_r50"
    elif exp_base == "style": exp_name_imapp_50 = "imapp_style_r50"
    elif exp_base == "asym_style": exp_name_imapp_50 = "imapp_asym_style_r50"
    elif exp_base == "hybrid_moe": exp_name_imapp_50 = "imapp_hybrid_moe_r50"
        
    res_imapp_50 = ("--", "--", "--")
    csv_imapp_50 = '/path/to/output/evaluations_imapp_50/all_experiments_summary.csv'
    if os.path.exists(csv_imapp_50):
        df_i50 = pd.read_csv(csv_imapp_50)
        df_i50 = df_i50[df_i50['experiment'] == exp_name_imapp_50]
        if len(df_i50) > 0:
            row = df_i50.iloc[0]
            t_iou, t_iou_gap = row['test_true_iou']*100, row['test_true_iou_gap']*100
            exc, defc = row['cl_ler_clean']/1e6, row['cl_ler_biased']/1e6
            rr, sdr_gap = row['cl_rr'], row['cl_sdr_gap']*100
            rr_str = f"{rr:.2f}" if rr < 1000 else "\\infty"
            res_imapp_50 = (f"${t_iou:.1f}/{t_iou_gap:.1f}$", f"${exc:.2f}/{defc:.2f}$", f"${rr_str}/{sdr_gap:.1f}$")
        
    return res_50, res_100, res_phc, res_imapp, res_imapp_50

methods = [
    ("ERM", "baseline"),
    ("EO", "fairness_eo"),
    ("DP", "fairness_dp"),
    ("EO+DP", "fairness_both"),
    ("Adversarial", "adversarial"),
    ("MMD", "mmd"),
    ("CORAL", "coral"),
    ("GCE", "gce"),
    ("Bootstrapping", "bootstrapping"),
    ("Asym. mask", "asym"),
    ("$g$-conditioned", "style"),
    ("Combined", "asym_style"),
    ("Auto-conditioned", "hybrid_moe")
]

# Legacy overrides for ERM
erm_50 = ("$93.1/5.5$", "$39.3/86.4$", "$4.68/-6.7$")

lines = []
for name, base in methods:
    if base == "baseline":
        r50 = erm_50
        r100 = ("--", "--", "--")
        rphc = ("--", "--", "--")
        rimapp_50 = ("--", "--", "--")
        
        csv_imapp_50 = '/path/to/output/evaluations_imapp_50/all_experiments_summary.csv'
        if os.path.exists(csv_imapp_50):
            df_i50 = pd.read_csv(csv_imapp_50)
            df_i50 = df_i50[df_i50['experiment'] == 'imapp_baseline_r50']
            if len(df_i50) > 0:
                row = df_i50.iloc[0]
                t_iou, t_iou_gap = row['test_true_iou']*100, row['test_true_iou_gap']*100
                exc, defc = row['cl_ler_clean']/1e6, row['cl_ler_biased']/1e6
                rr, sdr_gap = row['cl_rr'], row['cl_sdr_gap']*100
                rr_str = f"{rr:.2f}" if rr < 1000 else "\\infty"
                rimapp_50 = (f"${t_iou:.1f}/{t_iou_gap:.1f}$", f"${exc:.2f}/{defc:.2f}$", f"${rr_str}/{sdr_gap:.1f}$")
    else:
        r50, r100, rphc, _, rimapp_50 = get_row_data(base)
    
    if name == "Combined":
        r50 = (f"\\mathbf{{{r50[0].replace('$', '')}}}", r50[1], r50[2])
    
    line = f"& {name} & {r50[0]} & {r50[1]} & {r50[2]} & {r100[0]} & {r100[1]} & {r100[2]} & {rphc[0]} & {rphc[1]} & {rphc[2]} & {rimapp_50[0]} & {rimapp_50[1]} & {rimapp_50[2]} \\\\"
    lines.append(line)

print("\n".join(lines))
