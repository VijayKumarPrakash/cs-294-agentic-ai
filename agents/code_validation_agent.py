"""
Code Validation Agent for Python Code.

This agent validates Python code for correctness, quality, and effectiveness,
ensuring syntax validity, best practices adherence, functional correctness, and readability.
"""

import ast
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from agents.a2a_protocol import A2AProtocolHandler, A2AMessage, MessageStatus


class CodeValidationAgent:
    """
    Sub-agent responsible for validating Python code.

    Performs checks on:
    1. Syntax validation
    2. Best practices and PEP 8 style
    3. Functional correctness
    4. Readability and documentation
    """

    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.0):
        """
        Initialize the code validation agent.

        Args:
            model: The Gemini model to use
            temperature: Temperature for LLM responses
        """
        self.agent_id = "code_validation_agent"
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
        self.a2a_handler = A2AProtocolHandler(self.agent_id)

    def process_request(self, request: A2AMessage) -> A2AMessage:
        """
        Process an A2A request for code validation.

        Args:
            request: A2A request message from orchestrator

        Returns:
            A2A response message with validation results
        """
        try:
            # Extract validation context
            data = request.data or {}
            code_path = data.get("code_path", "")
            code = data.get("code", "")
            description = data.get("description", "")
            expected_behavior = data.get("expected_behavior", "")

            # Load code from file if code_path is provided
            if code_path and not code:
                code = self._load_code_from_file(code_path)
                print(f"[{self.agent_id.upper()}] Processing code validation request")
                print(f"[{self.agent_id.upper()}] Code file: {code_path}")
                print(f"[{self.agent_id.upper()}] Code length: {len(code)} characters")
            else:
                print(f"[{self.agent_id.upper()}] Processing code validation request")
                print(f"[{self.agent_id.upper()}] Code length: {len(code)} characters")

            if description:
                print(f"[{self.agent_id.upper()}] Description: {description}\n")
            else:
                print()

            # Perform validation checks
            validation_results = self._perform_validation(
                code=code,
                description=description,
                expected_behavior=expected_behavior
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

    def _load_code_from_file(self, file_path: str) -> str:
        """
        Load Python code from a file.

        Args:
            file_path (str): Path to the Python file

        Returns:
            str: Contents of the Python file

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file is not a Python file
        """
        import os

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Code file not found: {file_path}")

        if not file_path.endswith('.py'):
            raise ValueError(f"File must be a Python file (.py): {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            return code
        except Exception as e:
            raise ValueError(f"Error reading code file {file_path}: {str(e)}")

    def _perform_validation(
        self,
        code: str,
        description: str,
        expected_behavior: str
    ) -> Dict[str, Any]:
        """
        Perform comprehensive code validation.

        Args:
            code: Python code to validate
            description: Description of what the code should do
            expected_behavior: Expected behavior or output

        Returns:
            Dictionary with validation results including scores
        """
        results = {
            "checks_passed": [],
            "checks_failed": [],
            "warnings": [],
            "overall_status": "unknown",
            "details": {},
            "syntax_score": 0,
            "best_practices_score": 0,
            "functionality_score": 0,
            "readability_score": 0,
            "overall_score": 0.0,
            "feedback": {}
        }

        # Check 1: Syntax Validation
        syntax_check = self._check_syntax(code)
        results["syntax_score"] = syntax_check["score"]
        results["feedback"]["syntax"] = syntax_check["feedback"]
        if syntax_check["passed"]:
            results["checks_passed"].append("syntax_validation")
        else:
            results["checks_failed"].append("syntax_validation")
        results["details"]["syntax_validation"] = syntax_check

        # Check 2: Best Practices & PEP 8
        best_practices_check = self._check_best_practices(code)
        results["best_practices_score"] = best_practices_check["score"]
        results["feedback"]["best_practices"] = best_practices_check["feedback"]
        if best_practices_check["passed"]:
            results["checks_passed"].append("best_practices")
        else:
            results["checks_failed"].append("best_practices")
        results["details"]["best_practices"] = best_practices_check

        # Check 3: Functional Correctness
        functionality_check = self._check_functionality(code, description, expected_behavior)
        results["functionality_score"] = functionality_check["score"]
        results["feedback"]["functionality"] = functionality_check["feedback"]
        if functionality_check["passed"]:
            results["checks_passed"].append("functional_correctness")
        else:
            results["checks_failed"].append("functional_correctness")
        results["details"]["functional_correctness"] = functionality_check

        # Check 4: Readability & Documentation
        readability_check = self._check_readability(code)
        results["readability_score"] = readability_check["score"]
        results["feedback"]["readability"] = readability_check["feedback"]
        if readability_check["passed"]:
            results["checks_passed"].append("readability_documentation")
        else:
            results["checks_failed"].append("readability_documentation")
        results["details"]["readability_documentation"] = readability_check

        # Calculate overall score
        results["overall_score"] = round(
            (results["syntax_score"] +
             results["best_practices_score"] +
             results["functionality_score"] +
             results["readability_score"]) / 4.0,
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

    def _check_syntax(self, code: str) -> Dict[str, Any]:
        """
        Check if code is syntactically correct using AST parsing and LLM evaluation.
        """
        print(f"[{self.agent_id.upper()}] Running Check 1: Syntax Validation")

        try:
            # First, try to parse with AST
            ast.parse(code)
            syntax_error = None
            can_parse = True
        except SyntaxError as e:
            syntax_error = str(e)
            can_parse = False

        try:
            # Use LLM to evaluate syntax
            prompt = f"""Evaluate the syntax correctness of this Python code.

Python Code:
```python
{code}
```

AST Parsing Result: {"✓ Successfully parsed (no syntax errors)" if can_parse else f"✗ Syntax Error: {syntax_error}"}

Assess the syntax validity and provide:
1. A score from 0-10 (10 = perfect syntax, 0 = multiple critical errors)
2. Specific feedback about syntax issues or confirmation that syntax is correct
3. If there are errors, list them clearly

Format your response as:
SCORE: [0-10]
FEEDBACK: [Your detailed feedback]
"""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content

            # Parse response
            score = self._extract_score(content)
            feedback = self._extract_feedback(content)

            return {
                "passed": score >= 8,
                "score": score,
                "feedback": feedback,
                "analysis": content,
                "can_parse": can_parse,
                "syntax_error": syntax_error
            }

        except Exception as e:
            return {
                "passed": False,
                "score": 0,
                "feedback": f"Could not validate syntax: {str(e)}",
                "error": str(e)
            }

    def _check_best_practices(self, code: str) -> Dict[str, Any]:
        """
        Check if code follows Python best practices and PEP 8 style guidelines.
        """
        print(f"[{self.agent_id.upper()}] Running Check 2: Best Practices & PEP 8")

        try:
            prompt = f"""Evaluate this Python code for adherence to best practices and PEP 8 style guidelines.

Python Code:
```python
{code}
```

Assess the following aspects:
1. **PEP 8 Compliance**: Indentation (4 spaces), line length (<79 chars), spacing, naming conventions
2. **Naming Conventions**:
   - Functions/variables: lowercase_with_underscores
   - Classes: CapitalizedWords
   - Constants: UPPERCASE_WITH_UNDERSCORES
3. **Code Structure**: Proper imports, logical organization, no unnecessary complexity
4. **Python Idioms**: Use of list comprehensions where appropriate, context managers, etc.
5. **Code Smells**: Redundant code, magic numbers, overly complex logic

Provide:
1. A score from 0-10 (10 = exemplary practices, 0 = poor practices)
2. Specific feedback about what's good and what needs improvement

Format your response as:
SCORE: [0-10]
FEEDBACK: [Your detailed feedback with specific examples]
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
                "feedback": f"Could not check best practices: {str(e)}",
                "error": str(e)
            }

    def _check_functionality(
        self, code: str, description: str, expected_behavior: str
    ) -> Dict[str, Any]:
        """
        Check if code fulfills its intended purpose and implements correct functionality.
        """
        print(f"[{self.agent_id.upper()}] Running Check 3: Functional Correctness")

        try:
            context_info = ""
            if description:
                context_info += f"\n**Code Purpose**: {description}"
            if expected_behavior:
                context_info += f"\n**Expected Behavior**: {expected_behavior}"

            prompt = f"""Evaluate the functional correctness of this Python code.

Python Code:
```python
{code}
```
{context_info if context_info else "\n**Note**: No description provided. Infer purpose from code."}

Assess the following:
1. **Logic Correctness**: Does the code logic make sense? Are there any logical errors?
2. **Purpose Fulfillment**: Does the code achieve what it's supposed to do?
3. **Edge Cases**: Does the code handle edge cases appropriately?
4. **Potential Bugs**: Are there any potential runtime errors or bugs?
5. **Completeness**: Is the implementation complete or are there missing parts?

Reason through the code execution step by step if necessary.

Provide:
1. A score from 0-10 (10 = perfectly correct and complete, 0 = fundamentally broken)
2. Specific feedback about correctness, potential issues, and improvements

Format your response as:
SCORE: [0-10]
FEEDBACK: [Your detailed analysis of functional correctness]
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
                "feedback": f"Could not check functionality: {str(e)}",
                "error": str(e)
            }

    def _check_readability(self, code: str) -> Dict[str, Any]:
        """
        Check code readability, maintainability, and documentation quality.
        """
        print(f"[{self.agent_id.upper()}] Running Check 4: Readability & Documentation")

        try:
            prompt = f"""Evaluate the readability and documentation quality of this Python code.

Python Code:
```python
{code}
```

Assess the following aspects:
1. **Variable Naming**: Are variable names meaningful and descriptive?
2. **Function/Class Names**: Do names clearly convey purpose?
3. **Comments**: Are there helpful comments where needed? Too many or too few?
4. **Docstrings**: Are functions/classes documented with proper docstrings?
5. **Code Clarity**: Is the code easy to understand? Could it be simplified?
6. **Maintainability**: Would another developer easily understand and modify this code?

Provide:
1. A score from 0-10 (10 = excellently documented and readable, 0 = unreadable)
2. Specific feedback about what helps or hinders readability

Format your response as:
SCORE: [0-10]
FEEDBACK: [Your detailed feedback on readability and documentation]
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
                "feedback": f"Could not check readability: {str(e)}",
                "error": str(e)
            }

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
