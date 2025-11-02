"""
Entry point for the multi-agent validation system.

This script demonstrates how to use the orchestrating agent
to coordinate validation tasks using sub-agents via A2A protocol.
"""

import os
from dotenv import load_dotenv
from agents.orchestrator import OrchestratingAgent
from agents.state import ABTestContext, CodeValidationContext


def main():
    """Run A/B testing validation through the orchestrating agent."""

    # Load environment variables
    load_dotenv()

    # Verify API key is set
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        print("Please copy .env.example to .env and add your Google API key.")
        print("Get your API key from: https://makersuite.google.com/app/apikey")
        return

    print("=" * 70)
    print("Multi-Agent A/B Testing Validation System")
    print("=" * 70)
    print()

    # Initialize the orchestrating agent
    orchestrator = OrchestratingAgent(
        model="gemini-2.5-flash",
        temperature=0.0
    )

    # Define A/B test context
    ab_test_context = ABTestContext(
        hypothesis="Adding a premium tier pricing option will increase average revenue per user by 20%",
        success_metrics=["conversion", "revenue", "session_duration"],
        dataset_path="datasource/data/sample_ab_test.csv",
        expected_effect_size=0.3,  # Medium effect size
        significance_level=0.05,
        power=0.8
    )

    # Validation task
    task = """
    Validate the A/B test setup for a new premium pricing tier experiment.

    The test will show half of users (treatment group) a new premium pricing option
    at $79.99/month, while the control group sees the standard pricing at $49.99/month.

    We need to ensure:
    1. The dataset contains valid user data with proper group assignments
    2. Success metrics can be properly measured
    3. Sample size is sufficient for detecting the expected effect
    4. Data quality is appropriate for A/B testing analysis
    """

    print(f"Task: {task}\n")
    print("=" * 70)
    print()

    # Execute validation
    result = orchestrator.validate(task, ab_test_context=ab_test_context)

    # Display results
    print("=" * 70)
    print("FINAL VALIDATION RESULTS")
    print("=" * 70)
    print(f"\nValidation Status: {'✓ PASSED' if result['validation_passed'] else '✗ FAILED'}")
    print(f"\nSummary:\n{result['summary']}")

    # Show data validation details
    if result.get('data_validation'):
        print("\n" + "=" * 70)
        print("DATA VALIDATION DETAILS")
        print("=" * 70)
        data_val = result['data_validation']
        print(f"\nOverall Status: {data_val.get('overall_status', 'unknown').upper()}")
        print(f"Checks Passed: {len(data_val.get('checks_passed', []))}")
        print(f"Checks Failed: {len(data_val.get('checks_failed', []))}")

        if data_val.get('checks_failed'):
            print(f"\nFailed Checks: {', '.join(data_val['checks_failed'])}")

    print("\n" + "=" * 70)

    # Code Validation
    print("\n\n")
    print("=" * 70)
    print("Python Code Validation")
    print("=" * 70)
    print()

    code_validation_context = CodeValidationContext(
        code_path="datasource/sample_python_files/syntax_errors_ab_test.py",
        code=None,
        description="A/B testing implementation with statistical tests",
        expected_behavior="Should perform t-tests and chi-square tests on A/B test data"
    )

    code_task = "Validate the Python code quality for the A/B testing implementation"

    print(f"Task: {code_task}")
    print(f"File: datasource/sample_python_files/syntax_errors_ab_test.py\n")
    print("=" * 70)
    print()

    # Execute code validation
    code_result = orchestrator.validate(code_task, code_validation_context=code_validation_context)

    # Display results
    print("=" * 70)
    print("FINAL CODE VALIDATION RESULTS")
    print("=" * 70)
    print(f"\nValidation Status: {'✓ PASSED' if code_result['validation_passed'] else '✗ FAILED'}")
    print(f"\nSummary:\n{code_result['summary']}")

    # Show code validation details
    if code_result.get('code_validation'):
        print("\n" + "=" * 70)
        print("CODE VALIDATION DETAILS")
        print("=" * 70)
        code_val = code_result['code_validation']
        print(f"\nOverall Status: {code_val.get('overall_status', 'unknown').upper()}")
        print(f"Overall Score: {code_val.get('overall_score', 0)}/10")
        print(f"\nIndividual Scores:")
        print(f"  - Syntax:         {code_val.get('syntax_score', 0)}/10")
        print(f"  - Best Practices: {code_val.get('best_practices_score', 0)}/10")
        print(f"  - Functionality:  {code_val.get('functionality_score', 0)}/10")
        print(f"  - Readability:    {code_val.get('readability_score', 0)}/10")

        if code_val.get('checks_passed'):
            print(f"\nChecks Passed: {', '.join(code_val['checks_passed'])}")
        if code_val.get('checks_failed'):
            print(f"Checks Failed: {', '.join(code_val['checks_failed'])}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
