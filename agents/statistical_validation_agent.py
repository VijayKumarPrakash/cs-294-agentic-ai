"""
Statistical Test Validation Agent for A/B Testing.

This agent validates statistical tests and analyses for A/B testing experiments,
ensuring statistical rigor, appropriate test selection, and valid interpretations.
"""

import os
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from agents.a2a_protocol import A2AProtocolHandler, A2AMessage, MessageStatus


class StatisticalValidationAgent:
    """
    Sub-agent responsible for validating statistical tests in A/B testing.

    Performs checks on:
    1. Test selection and appropriateness
    2. Statistical assumptions validation
    3. Power analysis and sample size adequacy
    4. Effect size and practical significance
    5. P-value interpretation and significance testing
    6. Confidence intervals validity
    """

    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.0):
        """
        Initialize the statistical validation agent.

        Args:
            model: The Gemini model to use
            temperature: Temperature for LLM responses
        """
        self.agent_id = "statistical_validation_agent"
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
        self.a2a_handler = A2AProtocolHandler(self.agent_id)

    def process_request(self, request: A2AMessage) -> A2AMessage:
        """
        Process an A2A request for statistical validation.

        Args:
            request: A2A request message from orchestrator

        Returns:
            A2A response message with validation results
        """
        try:
            # Extract validation context
            data = request.data or {}
            test_results_path = data.get("test_results_path", "")
            test_results = data.get("test_results", "")
            dataset_path = data.get("dataset_path", "")
            test_type = data.get("test_type", "")
            hypothesis = data.get("hypothesis", "")
            significance_level = data.get("significance_level", 0.05)

            # Load test results from file if path is provided
            if test_results_path and not test_results:
                test_results = self._load_test_results_from_file(test_results_path)
                print(f"[{self.agent_id.upper()}] Processing statistical validation request")
                print(f"[{self.agent_id.upper()}] Test results file: {test_results_path}")
            else:
                print(f"[{self.agent_id.upper()}] Processing statistical validation request")

            if test_type:
                print(f"[{self.agent_id.upper()}] Test type: {test_type}")
            if hypothesis:
                print(f"[{self.agent_id.upper()}] Hypothesis: {hypothesis}\n")
            else:
                print()

            # Perform validation checks
            validation_results = self._perform_validation(
                test_results=test_results,
                dataset_path=dataset_path,
                test_type=test_type,
                hypothesis=hypothesis,
                significance_level=significance_level
            )

            print(f"[{self.agent_id.upper()}] Validation complete\n")

            # Create response
            response = self.a2a_handler.create_response(
                request_message=request,
                result=validation_results,
                status=MessageStatus.COMPLETED
            )

            return response

        except Exception as e:
            print(f"[{self.agent_id.upper()}] Error: {str(e)}\n")
            return self.a2a_handler.create_response(
                request_message=request,
                result={},
                status=MessageStatus.FAILED,
                error=str(e)
            )

    def _load_test_results_from_file(self, file_path: str) -> str:
        """
        Load statistical test results from a file.

        Args:
            file_path: Path to the test results file

        Returns:
            str: Contents of the test results file

        Raises:
            FileNotFoundError: If the file does not exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Test results file not found: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            raise ValueError(f"Error reading test results file {file_path}: {str(e)}")

    def _perform_validation(
        self,
        test_results: str,
        dataset_path: str,
        test_type: str,
        hypothesis: str,
        significance_level: float
    ) -> Dict[str, Any]:
        """
        Perform comprehensive statistical validation.

        Args:
            test_results: Statistical test results content
            dataset_path: Path to dataset (optional)
            test_type: Type of statistical test
            hypothesis: Hypothesis being tested
            significance_level: Significance level (alpha)

        Returns:
            Dictionary with validation results including scores
        """
        results = {
            "checks_passed": [],
            "checks_failed": [],
            "warnings": [],
            "overall_status": "unknown",
            "details": {},
            "test_selection_score": 0,
            "assumptions_score": 0,
            "power_analysis_score": 0,
            "effect_size_score": 0,
            "pvalue_score": 0,
            "confidence_interval_score": 0,
            "overall_score": 0.0,
            "feedback": {}
        }

        # Load dataset info if available
        dataset_info = None
        if dataset_path and os.path.exists(dataset_path):
            dataset_info = self._get_dataset_info(dataset_path)

        # Check 1: Test Selection & Appropriateness
        test_selection_check = self._check_test_selection(
            test_results, test_type, hypothesis, dataset_info
        )
        results["test_selection_score"] = test_selection_check["score"]
        results["feedback"]["test_selection"] = test_selection_check["feedback"]
        if test_selection_check["passed"]:
            results["checks_passed"].append("test_selection")
        else:
            results["checks_failed"].append("test_selection")
        results["details"]["test_selection"] = test_selection_check

        # Check 2: Assumptions Validation
        assumptions_check = self._check_assumptions(
            test_results, test_type, dataset_info
        )
        results["assumptions_score"] = assumptions_check["score"]
        results["feedback"]["assumptions"] = assumptions_check["feedback"]
        if assumptions_check["passed"]:
            results["checks_passed"].append("assumptions_validation")
        else:
            results["checks_failed"].append("assumptions_validation")
        results["details"]["assumptions_validation"] = assumptions_check

        # Check 3: Power Analysis & Sample Size
        power_check = self._check_power_analysis(
            test_results, dataset_info, significance_level
        )
        results["power_analysis_score"] = power_check["score"]
        results["feedback"]["power_analysis"] = power_check["feedback"]
        if power_check["passed"]:
            results["checks_passed"].append("power_analysis")
        else:
            results["checks_failed"].append("power_analysis")
        results["details"]["power_analysis"] = power_check

        # Check 4: Effect Size & Practical Significance
        effect_size_check = self._check_effect_size(
            test_results, test_type
        )
        results["effect_size_score"] = effect_size_check["score"]
        results["feedback"]["effect_size"] = effect_size_check["feedback"]
        if effect_size_check["passed"]:
            results["checks_passed"].append("effect_size")
        else:
            results["checks_failed"].append("effect_size")
        results["details"]["effect_size"] = effect_size_check

        # Check 5: P-Value & Significance Testing
        pvalue_check = self._check_pvalue_interpretation(
            test_results, significance_level
        )
        results["pvalue_score"] = pvalue_check["score"]
        results["feedback"]["pvalue"] = pvalue_check["feedback"]
        if pvalue_check["passed"]:
            results["checks_passed"].append("pvalue_interpretation")
        else:
            results["checks_failed"].append("pvalue_interpretation")
        results["details"]["pvalue_interpretation"] = pvalue_check

        # Check 6: Confidence Intervals
        ci_check = self._check_confidence_intervals(
            test_results, test_type
        )
        results["confidence_interval_score"] = ci_check["score"]
        results["feedback"]["confidence_intervals"] = ci_check["feedback"]
        if ci_check["passed"]:
            results["checks_passed"].append("confidence_intervals")
        else:
            results["checks_failed"].append("confidence_intervals")
        results["details"]["confidence_intervals"] = ci_check

        # Calculate overall score
        results["overall_score"] = round(
            (results["test_selection_score"] +
             results["assumptions_score"] +
             results["power_analysis_score"] +
             results["effect_size_score"] +
             results["pvalue_score"] +
             results["confidence_interval_score"]) / 6.0,
            1
        )

        # Determine overall status
        if results["overall_score"] >= 8.0:
            results["overall_status"] = "passed"
        elif results["overall_score"] >= 6.0:
            results["overall_status"] = "passed_with_warnings"
        else:
            results["overall_status"] = "failed"

        return results

    def _check_test_selection(
        self, test_results: str, test_type: str, hypothesis: str, dataset_info: Optional[Dict]
    ) -> Dict[str, Any]:
        """
        Check 1: Test Selection & Appropriateness
        Validate that the chosen test is appropriate for the data and hypothesis.
        """
        print(f"[{self.agent_id.upper()}] Running Check 1: Test Selection & Appropriateness")

        try:
            dataset_context = ""
            if dataset_info:
                dataset_context = f"""
Dataset Information:
- Sample size: {dataset_info.get('sample_size', 'Unknown')}
- Numeric columns: {dataset_info.get('numeric_columns', [])}
- Categorical columns: {dataset_info.get('categorical_columns', [])}
"""

            prompt = f"""Evaluate whether the statistical test selection is appropriate for this A/B testing scenario.

Test Results:
{test_results}

Test Type: {test_type if test_type else "Not specified - infer from results"}
Hypothesis: {hypothesis if hypothesis else "Not specified - infer from results"}
{dataset_context}

Assess the appropriateness of test selection based on:
1. **Data Type Match**: Does the test match the data types (continuous, categorical, ordinal)?
2. **Hypothesis Alignment**: Does the test properly address the stated hypothesis?
3. **Comparison Type**: Is the test suitable for the number of groups and comparison structure?
4. **Common Misapplications**: Are there signs of inappropriate test usage (e.g., t-test on non-normal small samples)?
5. **Alternative Tests**: Would a different test be more appropriate?

Guidelines:
- T-tests for comparing means of two groups (continuous data)
- Chi-square for categorical data associations
- ANOVA for comparing means across multiple groups
- Mann-Whitney U/Wilcoxon for non-parametric comparisons
- Proportion tests (z-test) for comparing proportions

Provide:
1. Score from 0-10 (10 = perfect test selection, 0 = inappropriate test)
2. Detailed feedback on test appropriateness
3. Alternative recommendations if applicable

Format your response as:
SCORE: [0-10]
FEEDBACK: [Your detailed analysis of test selection appropriateness]
"""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content

            # Parse response
            score = self._extract_score(content)
            feedback = self._extract_feedback(content)

            return {
                "passed": score >= 7,
                "score": score,
                "feedback": feedback,
                "analysis": content
            }

        except Exception as e:
            return {
                "passed": False,
                "score": 0,
                "feedback": f"Could not validate test selection: {str(e)}",
                "error": str(e)
            }

    def _check_assumptions(
        self, test_results: str, test_type: str, dataset_info: Optional[Dict]
    ) -> Dict[str, Any]:
        """
        Check 2: Assumptions Validation
        Evaluate whether statistical test assumptions are met or properly addressed.
        """
        print(f"[{self.agent_id.upper()}] Running Check 2: Assumptions Validation")

        try:
            # Calculate basic statistics if dataset is available
            assumptions_context = ""
            if dataset_info:
                assumptions_context = f"""
Dataset Characteristics:
- Sample size: {dataset_info.get('sample_size', 'Unknown')}
- Distribution indicators: {dataset_info.get('distribution_notes', 'Not calculated')}
"""

            prompt = f"""Evaluate whether the statistical test assumptions are properly validated and met.

Test Results:
{test_results}

Test Type: {test_type if test_type else "Infer from results"}
{assumptions_context}

Assess the following assumptions based on the test type:

For T-tests/ANOVA:
1. **Normality**: Is the data approximately normally distributed? (especially critical for small samples)
2. **Homogeneity of Variance**: Are variances equal across groups?
3. **Independence**: Are observations independent?
4. **Sample Size**: Is sample size adequate for Central Limit Theorem to apply?

For Chi-square tests:
1. **Expected Frequencies**: Are expected frequencies ≥5 in each cell?
2. **Independence**: Are observations independent?

For Non-parametric tests:
1. **Independence**: Are observations independent?
2. **Similar Distributions**: Do groups have similar distribution shapes?

Evaluate:
- Are assumptions explicitly checked and reported?
- Are violations acknowledged and addressed?
- Are appropriate corrections or alternative tests used when assumptions fail?
- Is there evidence of proper randomization?

Provide:
1. Score from 0-10 (10 = all assumptions met/properly handled, 0 = critical violations ignored)
2. Detailed feedback on assumption validation

Format your response as:
SCORE: [0-10]
FEEDBACK: [Your detailed analysis of assumptions]
"""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content

            # Parse response
            score = self._extract_score(content)
            feedback = self._extract_feedback(content)

            return {
                "passed": score >= 6,
                "score": score,
                "feedback": feedback,
                "analysis": content
            }

        except Exception as e:
            return {
                "passed": False,
                "score": 0,
                "feedback": f"Could not validate assumptions: {str(e)}",
                "error": str(e)
            }

    def _check_power_analysis(
        self, test_results: str, dataset_info: Optional[Dict], significance_level: float
    ) -> Dict[str, Any]:
        """
        Check 3: Power Analysis & Sample Size
        Assess whether power analysis was conducted and sample size is adequate.
        """
        print(f"[{self.agent_id.upper()}] Running Check 3: Power Analysis & Sample Size")

        try:
            sample_context = ""
            if dataset_info:
                sample_size = dataset_info.get('sample_size', 0)
                # Calculate theoretical minimum sample size for medium effect (Cohen's d = 0.5)
                # Using simplified power analysis for context
                from scipy.stats import norm
                z_alpha = norm.ppf(1 - significance_level / 2)
                z_beta = norm.ppf(0.8)  # 80% power
                effect_size = 0.5  # medium effect
                required_n = ((z_alpha + z_beta) ** 2) * 2 / (effect_size ** 2)

                sample_context = f"""
Sample Size Information:
- Actual total sample size: {sample_size}
- Theoretical minimum for medium effect (d=0.5): ~{int(required_n * 2)} total
- Alpha: {significance_level}
- Target power (standard): 0.80 (80%)
"""

            prompt = f"""Evaluate the power analysis and sample size adequacy for this statistical test.

Test Results:
{test_results}
{sample_context}

Assess:
1. **Power Analysis Presence**: Was a power analysis conducted before or after the test?
2. **Sample Size Adequacy**: Is the sample size sufficient to detect meaningful effects?
3. **Effect Size Consideration**: Is the expected or observed effect size discussed?
4. **Type II Error Risk**: Is the risk of false negatives (failing to detect real effects) addressed?
5. **Minimum Detectable Effect**: Is the minimum detectable effect (MDE) reported?
6. **Power Calculation**: If power is reported, is it calculated correctly?

Guidelines:
- Standard power target: 0.80 (80%)
- Minimum sample sizes vary by effect size (small effects need larger samples)
- Post-hoc power analysis has limitations but can inform future studies
- Underpowered studies risk missing real effects

Provide:
1. Score from 0-10 (10 = proper power analysis with adequate sample, 0 = no consideration of power)
2. Detailed feedback on power and sample size

Format your response as:
SCORE: [0-10]
FEEDBACK: [Your detailed analysis of power and sample size]
"""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content

            # Parse response
            score = self._extract_score(content)
            feedback = self._extract_feedback(content)

            return {
                "passed": score >= 6,
                "score": score,
                "feedback": feedback,
                "analysis": content
            }

        except Exception as e:
            return {
                "passed": False,
                "score": 0,
                "feedback": f"Could not validate power analysis: {str(e)}",
                "error": str(e)
            }

    def _check_effect_size(
        self, test_results: str, test_type: str
    ) -> Dict[str, Any]:
        """
        Check 4: Effect Size & Practical Significance
        Evaluate whether effect sizes are reported and practically significant.
        """
        print(f"[{self.agent_id.upper()}] Running Check 4: Effect Size & Practical Significance")

        try:
            prompt = f"""Evaluate the effect size reporting and practical significance assessment.

Test Results:
{test_results}

Test Type: {test_type if test_type else "Infer from results"}

Assess:
1. **Effect Size Metrics**: Are appropriate effect size metrics reported?
   - Cohen's d (for t-tests): Small (0.2), Medium (0.5), Large (0.8)
   - Pearson's r (correlations): Small (0.1), Medium (0.3), Large (0.5)
   - Eta-squared/Omega-squared (ANOVA)
   - Odds ratio, Risk ratio (categorical outcomes)
   - Percentage differences (for proportions)

2. **Practical Significance**: Is the effect size practically meaningful beyond statistical significance?
3. **Business Context**: Is the effect size interpreted in terms of business impact or real-world relevance?
4. **Statistical vs Practical**: Is there discussion of statistical significance vs practical significance?
5. **Effect Size Interpretation**: Is the magnitude properly contextualized (small/medium/large)?

Important:
- A statistically significant result with tiny effect size may not be practically useful
- Large samples can make tiny, irrelevant differences statistically significant
- Effect sizes provide standardized measures of magnitude

Provide:
1. Score from 0-10 (10 = excellent effect size reporting and interpretation, 0 = no effect size discussed)
2. Detailed feedback on effect size and practical significance

Format your response as:
SCORE: [0-10]
FEEDBACK: [Your detailed analysis of effect size and practical significance]
"""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content

            # Parse response
            score = self._extract_score(content)
            feedback = self._extract_feedback(content)

            return {
                "passed": score >= 6,
                "score": score,
                "feedback": feedback,
                "analysis": content
            }

        except Exception as e:
            return {
                "passed": False,
                "score": 0,
                "feedback": f"Could not validate effect size: {str(e)}",
                "error": str(e)
            }

    def _check_pvalue_interpretation(
        self, test_results: str, significance_level: float
    ) -> Dict[str, Any]:
        """
        Check 5: P-Value & Significance Testing
        Validate p-value calculation and interpretation.
        """
        print(f"[{self.agent_id.upper()}] Running Check 5: P-Value & Significance Testing")

        try:
            prompt = f"""Evaluate the p-value calculation, reporting, and interpretation.

Test Results:
{test_results}

Significance Level (Alpha): {significance_level}

Assess:
1. **P-Value Calculation**: Is the p-value calculated correctly for the chosen test?
2. **Significance Level**: Is the alpha level (e.g., 0.05) clearly stated and justified?
3. **One-tailed vs Two-tailed**: Is the choice of one-tailed or two-tailed test appropriate and justified?
4. **P-Value Interpretation**: Is the p-value interpreted correctly?
   - Correct: "Probability of observing this result if null hypothesis is true"
   - Incorrect: "Probability that null hypothesis is true"

5. **P-Hacking Indicators**: Are there signs of questionable practices?
   - Multiple testing without correction (testing many metrics, reporting only significant ones)
   - P-values suspiciously close to 0.05
   - Selective reporting
   - Post-hoc hypothesis changes
   - Optional stopping (checking p-values repeatedly, stopping when significant)

6. **Borderline P-Values**: If p-value is close to threshold (0.045-0.055), is this acknowledged?
7. **Multiple Comparisons**: If multiple tests conducted, are corrections applied (Bonferroni, FDR, etc.)?

Guidelines:
- P < 0.05 doesn't mean "practically important" or "large effect"
- Non-significant doesn't mean "no effect" (could be underpowered)
- P-values are continuous, not binary cutoffs

Provide:
1. Score from 0-10 (10 = rigorous p-value analysis, 0 = misinterpretation or p-hacking)
2. Detailed feedback on p-value interpretation

Format your response as:
SCORE: [0-10]
FEEDBACK: [Your detailed analysis of p-value interpretation and practices]
"""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content

            # Parse response
            score = self._extract_score(content)
            feedback = self._extract_feedback(content)

            return {
                "passed": score >= 6,
                "score": score,
                "feedback": feedback,
                "analysis": content
            }

        except Exception as e:
            return {
                "passed": False,
                "score": 0,
                "feedback": f"Could not validate p-value interpretation: {str(e)}",
                "error": str(e)
            }

    def _check_confidence_intervals(
        self, test_results: str, test_type: str
    ) -> Dict[str, Any]:
        """
        Check 6: Confidence Intervals
        Validate confidence interval calculation and reporting.
        """
        print(f"[{self.agent_id.upper()}] Running Check 6: Confidence Intervals")

        try:
            prompt = f"""Evaluate the confidence interval calculation and interpretation.

Test Results:
{test_results}

Test Type: {test_type if test_type else "Infer from results"}

Assess:
1. **CI Presence**: Are confidence intervals reported for the estimates?
2. **CI Calculation**: Are CIs calculated correctly for the test type?
   - T-test: CI for mean difference
   - Proportion test: CI for proportion or difference in proportions
   - ANOVA: CIs for group means or pairwise differences
   - Regression: CIs for coefficients

3. **Confidence Level**: Is the confidence level clearly stated (typically 95%)?
4. **CI Width**: Is the width of the CI appropriate?
   - Narrow CI = precise estimate
   - Wide CI = imprecise estimate, may need larger sample

5. **Null Value**: Does the CI include the null value (e.g., 0 for differences)?
   - If CI excludes null → statistically significant
   - If CI includes null → not statistically significant

6. **Practical Interpretation**: Is the CI interpreted in practical terms?
7. **CI vs P-Value Consistency**: Are CI conclusions consistent with p-value results?

Guidelines:
- CIs provide more information than p-values alone (range of plausible values)
- Bootstrap CIs may be more appropriate for non-normal data
- CIs should be reported alongside point estimates

Provide:
1. Score from 0-10 (10 = excellent CI reporting, 0 = no CIs reported)
2. Detailed feedback on confidence intervals

Format your response as:
SCORE: [0-10]
FEEDBACK: [Your detailed analysis of confidence intervals]
"""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content

            # Parse response
            score = self._extract_score(content)
            feedback = self._extract_feedback(content)

            return {
                "passed": score >= 6,
                "score": score,
                "feedback": feedback,
                "analysis": content
            }

        except Exception as e:
            return {
                "passed": False,
                "score": 0,
                "feedback": f"Could not validate confidence intervals: {str(e)}",
                "error": str(e)
            }

    def _get_dataset_info(self, dataset_path: str) -> Dict[str, Any]:
        """Get information about the dataset for context in validation."""
        try:
            df = self._load_dataset(dataset_path)

            # Get basic info
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

            info = {
                "sample_size": len(df),
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "columns": list(df.columns)
            }

            return info

        except Exception as e:
            return {
                "error": str(e),
                "sample_size": 0
            }

    def _load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Load dataset from file."""
        if dataset_path.endswith('.csv'):
            return pd.read_csv(dataset_path)
        elif dataset_path.endswith('.json'):
            return pd.read_json(dataset_path)
        elif dataset_path.endswith('.parquet'):
            return pd.read_parquet(dataset_path)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")

    def _extract_score(self, content: str) -> int:
        """Extract score from LLM response."""
        try:
            for line in content.split("\n"):
                if "SCORE:" in line.upper():
                    # Extract number from line
                    score_str = line.split(":")[-1].strip()
                    # Handle cases like "8/10" or "8 out of 10"
                    if "/" in score_str or "out of" in score_str.lower():
                        score_str = score_str.split("/")[0].split("out")[0].strip()
                    # Extract first number found
                    import re
                    numbers = re.findall(r'\d+', score_str)
                    if numbers:
                        score = int(numbers[0])
                        return max(0, min(10, score))  # Clamp to 0-10
            return 5  # Default to 5 if no score found
        except:
            return 5

    def _extract_feedback(self, content: str) -> str:
        """Extract feedback from LLM response."""
        try:
            feedback_started = False
            feedback_lines = []
            for line in content.split("\n"):
                if "FEEDBACK:" in line.upper():
                    feedback_started = True
                    # Get the part after "FEEDBACK:"
                    feedback_lines.append(line.split(":", 1)[-1].strip())
                elif feedback_started:
                    feedback_lines.append(line)

            if feedback_lines:
                return " ".join(feedback_lines).strip()
            else:
                # Return full content if no FEEDBACK: marker found
                return content.strip()
        except:
            return content.strip()
