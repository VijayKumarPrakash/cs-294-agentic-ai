"""
A/B Test Statistical Analysis Module

This module provides functions for conducting A/B tests with proper
statistical analysis including t-tests, chi-square tests, and effect size calculations.
"""

from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
from scipy import stats


def load_ab_test_data(file_path: str) -> pd.DataFrame:
    """
    Load A/B test data from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing A/B test data

    Returns:
        pd.DataFrame: Loaded dataframe with A/B test data

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file format is invalid
    """
    try:
        data = pd.read_csv(file_path)
        required_columns = ['group', 'conversion', 'revenue']

        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")

        return data
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Data file not found: {file_path}") from e


def calculate_two_sample_ttest(
    control_data: np.ndarray,
    treatment_data: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform two-sample t-test for continuous metrics.

    Args:
        control_data (np.ndarray): Metric values for control group
        treatment_data (np.ndarray): Metric values for treatment group
        alpha (float): Significance level (default: 0.05)

    Returns:
        Dict[str, Any]: Dictionary containing test results with keys:
            - statistic: t-statistic value
            - p_value: p-value from the test
            - significant: boolean indicating statistical significance
            - control_mean: mean of control group
            - treatment_mean: mean of treatment group
            - effect_size: Cohen's d effect size

    Raises:
        ValueError: If input arrays are empty or have insufficient data
    """
    if len(control_data) == 0 or len(treatment_data) == 0:
        raise ValueError("Input arrays cannot be empty")

    if len(control_data) < 2 or len(treatment_data) < 2:
        raise ValueError("Need at least 2 samples per group for t-test")

    # Perform independent samples t-test
    t_statistic, p_value = stats.ttest_ind(control_data, treatment_data)

    # Calculate means
    control_mean = np.mean(control_data)
    treatment_mean = np.mean(treatment_data)

    # Calculate Cohen's d effect size
    pooled_std = np.sqrt(
        (np.var(control_data, ddof=1) + np.var(treatment_data, ddof=1)) / 2
    )
    cohens_d = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0

    return {
        'statistic': t_statistic,
        'p_value': p_value,
        'significant': p_value < alpha,
        'control_mean': control_mean,
        'treatment_mean': treatment_mean,
        'effect_size': cohens_d
    }


def calculate_chi_square_test(
    control_conversions: int,
    control_total: int,
    treatment_conversions: int,
    treatment_total: int,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform chi-square test for conversion rate analysis.

    Args:
        control_conversions (int): Number of conversions in control group
        control_total (int): Total users in control group
        treatment_conversions (int): Number of conversions in treatment group
        treatment_total (int): Total users in treatment group
        alpha (float): Significance level (default: 0.05)

    Returns:
        Dict[str, Any]: Dictionary containing test results with keys:
            - statistic: chi-square statistic
            - p_value: p-value from the test
            - significant: boolean indicating statistical significance
            - control_rate: conversion rate for control
            - treatment_rate: conversion rate for treatment
            - relative_uplift: percentage change in conversion

    Raises:
        ValueError: If totals are zero or negative
    """
    if control_total <= 0 or treatment_total <= 0:
        raise ValueError("Total counts must be positive")

    # Create contingency table
    contingency_table = np.array([
        [control_conversions, control_total - control_conversions],
        [treatment_conversions, treatment_total - treatment_conversions]
    ])

    # Perform chi-square test
    chi2_stat, p_value, _, _ = stats.chi2_contingency(contingency_table)

    # Calculate conversion rates
    control_rate = control_conversions / control_total
    treatment_rate = treatment_conversions / treatment_total

    # Calculate relative uplift
    relative_uplift = ((treatment_rate - control_rate) / control_rate * 100) if control_rate > 0 else 0

    return {
        'statistic': chi2_stat,
        'p_value': p_value,
        'significant': p_value < alpha,
        'control_rate': control_rate,
        'treatment_rate': treatment_rate,
        'relative_uplift': relative_uplift
    }


def run_complete_ab_test_analysis(
    data: pd.DataFrame,
    metric_column: str,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Run a complete A/B test analysis on the provided data.

    Args:
        data (pd.DataFrame): DataFrame with 'group' column and metric column
        metric_column (str): Name of the metric column to analyze
        alpha (float): Significance level (default: 0.05)

    Returns:
        Dict[str, Any]: Complete analysis results including descriptive stats
                       and test results

    Raises:
        ValueError: If required columns are missing or groups are invalid
    """
    if 'group' not in data.columns:
        raise ValueError("Data must contain 'group' column")

    if metric_column not in data.columns:
        raise ValueError(f"Metric column '{metric_column}' not found in data")

    # Split data by group
    control_data = data[data['group'] == 'control'][metric_column].values
    treatment_data = data[data['group'] == 'treatment'][metric_column].values

    if len(control_data) == 0 or len(treatment_data) == 0:
        raise ValueError("Both control and treatment groups must have data")

    # Run statistical test
    test_results = calculate_two_sample_ttest(control_data, treatment_data, alpha)

    # Add sample sizes
    test_results['control_size'] = len(control_data)
    test_results['treatment_size'] = len(treatment_data)

    return test_results


def main():
    """
    Main function demonstrating A/B test analysis workflow.
    """
    # Load data
    data_path = "datasource/data/sample_ab_test.csv"

    try:
        data = load_ab_test_data(data_path)
        print("✓ Data loaded successfully")
        print(f"  Total samples: {len(data)}")

        # Analyze revenue metric
        print("\n=== Revenue Analysis ===")
        revenue_results = run_complete_ab_test_analysis(data, 'revenue', alpha=0.05)

        print(f"Control Mean: ${revenue_results['control_mean']:.2f}")
        print(f"Treatment Mean: ${revenue_results['treatment_mean']:.2f}")
        print(f"Effect Size (Cohen's d): {revenue_results['effect_size']:.3f}")
        print(f"P-value: {revenue_results['p_value']:.4f}")
        print(f"Significant: {'Yes' if revenue_results['significant'] else 'No'}")

        # Analyze conversion rates
        print("\n=== Conversion Rate Analysis ===")
        control_conversions = data[data['group'] == 'control']['conversion'].sum()
        control_total = len(data[data['group'] == 'control'])
        treatment_conversions = data[data['group'] == 'treatment']['conversion'].sum()
        treatment_total = len(data[data['group'] == 'treatment'])

        conversion_results = calculate_chi_square_test(
            control_conversions, control_total,
            treatment_conversions, treatment_total
        )

        print(f"Control Rate: {conversion_results['control_rate']:.2%}")
        print(f"Treatment Rate: {conversion_results['treatment_rate']:.2%}")
        print(f"Relative Uplift: {conversion_results['relative_uplift']:.2f}%")
        print(f"P-value: {conversion_results['p_value']:.4f}")
        print(f"Significant: {'Yes' if conversion_results['significant'] else 'No'}")

    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
