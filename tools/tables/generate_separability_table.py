import pandas as pd
import numpy as np

# 1. Load PhC Results
df_phc = pd.read_csv('/path/to/output/evaluations_phc/all_experiments_summary.csv')

# 2. Load IMA++ Results
df_imapp = pd.read_csv('/path/to/output/evaluations_imapp/all_experiments_summary.csv')

def format_row(row, is_imapp=False):
    # IoU and gap
    iou = row['test_true_iou'] * 100
    gap = row['test_true_iou_gap'] * 100
    iou_str = f"{iou:.1f}/{gap:.1f}"

    # E/D
    excess = int(row['cl_total_excess']) / 1e6
    deficit = int(row['cl_total_deficit']) / 1e6
    ed_str = f"{excess:.1f}/{deficit:.1f}"
    
    # RR / SDR
    rr = row['cl_rr'] * 100
    sdr_gap = row['cl_sdr_gap'] * 100
    rr_str = f"{rr:.1f}/{sdr_gap:.1f}"
    
    return iou_str, ed_str, rr_str

phc_rows = []
# PhC bias levels
bias_map_phc = {
    'phc_biased_r0.00_tint0.10': '0\%',
    'phc_biased_r0.25_tint0.10': '25\%',
    'phc_biased_r0.50_tint0.10': '50\%',
    'phc_biased_r0.75_tint0.10': '75\%',
    'phc_biased_r1.0_tint0.10': '100\%'
}
for exp, label in bias_map_phc.items():
    row = df_phc[df_phc['experiment'] == exp]
    if not row.empty:
        r = row.iloc[0]
        iou, ed, rr = format_row(r, is_imapp=False)
        phc_rows.append(f"{label} & {iou} & {ed} & {rr} \\\\")

imapp_rows = []
# IMA++ bias levels
bias_map_imapp = {
    'imapp_bias_nobias': '0\%',
    'imapp_bias_50': '50\%',
    'imapp_bias_baseline': '100\%'
}
for exp, label in bias_map_imapp.items():
    row = df_imapp[df_imapp['experiment'] == exp]
    if not row.empty:
        r = row.iloc[0]
        iou, ed, rr = format_row(r, is_imapp=True)
        imapp_rows.append(f"{label} & {iou} & {ed} & {rr} \\\\")

latex = r"""\begin{table}[t]
\centering
\caption{
\textbf{Separability Analysis across Bias Levels.}
Evaluation on PhC-U373 and IMA++ datasets across varying levels of synthetic bias ($\beta$). 
$(\text{IoU}/\Delta)_{\text{true}}$ reports the mean True IoU and the subgroup performance gap in percentage. CL (E/D) reports the confident learning excess and deficit pixel counts (in millions). RR/SDR reports the Representational Rate and SDR gap ($\Delta$SDR).
}
\label{tab:separability_analysis}
\small
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Bias Level ($\beta$)} & \textbf{$(\text{IoU}/\Delta)_{\text{true}}$} & \textbf{CL (E/D)} & \textbf{RR/$\Delta$SDR} \\
\midrule
\multicolumn{4}{c}{\textbf{PhC-U373}} \\
\midrule
""" + "\n".join(phc_rows) + r"""
\midrule
\multicolumn{4}{c}{\textbf{IMA++}} \\
\midrule
""" + "\n".join(imapp_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

with open('separability_latex_table.md', 'w') as f:
    f.write("```latex\n" + latex + "\n```")

print("Generated table!")
