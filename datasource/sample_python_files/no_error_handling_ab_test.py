import pandas as pd
import numpy as np
from scipy import stats

def load_ab_test_data(file_path):
    data = pd.read_csv(file_path)
    return data

def calculate_ttest(control_data, treatment_data, alpha=0.05):
    t_statistic, p_value = stats.ttest_ind(control_data, treatment_data)

    control_mean = np.mean(control_data)
    treatment_mean = np.mean(treatment_data)

    pooled_std = np.sqrt(
        (np.var(control_data, ddof=1) + np.var(treatment_data, ddof=1)) / 2
    )

    cohens_d = (treatment_mean - control_mean) / pooled_std

    return {
        'statistic': t_statistic,
        'p_value': p_value,
        'significant': p_value < alpha,
        'control_mean': control_mean,
        'treatment_mean': treatment_mean,
        'effect_size': cohens_d
    }

def calculate_chi_square(control_conversions, control_total, treatment_conversions, treatment_total, alpha=0.05):
    contingency_table = np.array([
        [control_conversions, control_total - control_conversions],
        [treatment_conversions, treatment_total - treatment_conversions]
    ])

    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    control_rate = control_conversions / control_total
    treatment_rate = treatment_conversions / treatment_total

    relative_uplift = ((treatment_rate - control_rate) / control_rate) * 100

    return {
        'statistic': chi2_stat,
        'p_value': p_value,
        'significant': p_value < alpha,
        'control_rate': control_rate,
        'treatment_rate': treatment_rate,
        'relative_uplift': relative_uplift
    }

def run_ab_test_analysis(data, metric_column, alpha=0.05):
    control_data = data[data['group'] == 'control'][metric_column].values
    treatment_data = data[data['group'] == 'treatment'][metric_column].values

    test_results = calculate_ttest(control_data, treatment_data, alpha)

    test_results['control_size'] = len(control_data)
    test_results['treatment_size'] = len(treatment_data)

    return test_results

def main():
    data_path = "datasource/data/sample_ab_test.csv"

    data = load_ab_test_data(data_path)
    print("Data loaded successfully")
    print(f"Total samples: {len(data)}")

    print("\n=== Revenue Analysis ===")
    revenue_results = run_ab_test_analysis(data, 'revenue', alpha=0.05)

    print(f"Control Mean: ${revenue_results['control_mean']:.2f}")
    print(f"Treatment Mean: ${revenue_results['treatment_mean']:.2f}")
    print(f"Effect Size: {revenue_results['effect_size']:.3f}")
    print(f"P-value: {revenue_results['p_value']:.4f}")
    print(f"Significant: {'Yes' if revenue_results['significant'] else 'No'}")

    print("\n=== Conversion Rate Analysis ===")
    control_conversions = data[data['group'] == 'control']['conversion'].sum()
    control_total = len(data[data['group'] == 'control'])
    treatment_conversions = data[data['group'] == 'treatment']['conversion'].sum()
    treatment_total = len(data[data['group'] == 'treatment'])

    conversion_results = calculate_chi_square(
        control_conversions, control_total,
        treatment_conversions, treatment_total
    )

    print(f"Control Rate: {conversion_results['control_rate']:.2%}")
    print(f"Treatment Rate: {conversion_results['treatment_rate']:.2%}")
    print(f"Relative Uplift: {conversion_results['relative_uplift']:.2f}%")
    print(f"P-value: {conversion_results['p_value']:.4f}")
    print(f"Significant: {'Yes' if conversion_results['significant'] else 'No'}")

if __name__ == "__main__":
    main()
