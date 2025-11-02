import pandas as pd
import numpy as np
from scipy import stats

def load_data(file_path)
    """Load AB test data"""
    data = pd.read_csv(file_path)
    return data

def run_ttest(control, treatment, alpha=0.05):
    t_stat, p_val = stats.ttest_ind(control, treatment)

    control_mean = np.mean(control)
    treatment_mean = np.mean(treatment

    pooled_std = np.sqrt(
        (np.var(control, ddof=1) + np.var(treatment, ddof=1)) / 2
    )

    effect_size = (treatment_mean - control_mean) / pooled_std

    return {
        'p_value': p_val,
        'significant': p_val < alpha,
        'control_mean': control_mean
        'treatment_mean': treatment_mean,
        'effect_size': effect_size
    }

def run_chi_square(control_conv, control_total, treatment_conv, treatment_total):
    table = np.array([
        [control_conv, control_total - control_conv],
        [treatment_conv, treatment_total - treatment_conv]
    ])

    chi2, p_value, _, _ = stats.chi2_contingency(table)

    control_rate = control_conv / control_total
    treatment_rate = treatment_conv / treatment_total

    uplift = ((treatment_rate - control_rate) / control_rate) * 100

    return {
        'p_value': p_value,
        'control_rate': control_rate,
        'treatment_rate': treatment_rate
        'uplift': uplift
    ]

def analyze_ab_test(data):
    control_revenue = data[data['group'] == 'control']['revenue'].values
    treatment_revenue = data[data['group'] == 'treatment']['revenue'].values

    results = run_ttest(control_revenue, treatment_revenue)

    print(f"Control Mean: ${results['control_mean']:.2f}")
    print(f"Treatment Mean: ${results['treatment_mean']:.2f}"
    print(f"P-value: {results['p_value']:.4f}")

    return results

if __name__ == "__main__":
    data_path = "datasource/data/sample_ab_test.csv"
    data = load_data(data_path)

    results = analyze_ab_test(data)
    print("Analysis complete")
