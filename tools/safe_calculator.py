"""
Safe Mathematical Calculator for SmartStudy AI.
Uses Python's Abstract Syntax Tree (AST) evaluator with a strict node whitelist.
Prevents arbitrary code execution and rejects malicious inputs.
"""

import ast
import math
import operator
import re
from typing import Union, Any

# Supported safe operators
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Whitelisted mathematical functions
SAFE_FUNCTIONS = {
    "sqrt": math.sqrt,
    "pow": math.pow,
    "abs": abs,
    "round": round,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "factorial": math.factorial,
    "floor": math.floor,
    "ceil": math.ceil,
    "pi": math.pi,
    "e": math.e,
}


class CalculatorSecurityError(Exception):
    """Raised when an unsafe or unsupported expression is detected."""
    pass


def preprocess_expression(expr: str) -> str:
    """
    Clean and normalize user mathematical queries.
    Handles idioms like '25% of 480', '2^10', etc.
    """
    clean = expr.strip().rstrip("?.!;, ")

    # Pattern: X% of Y -> ((X / 100) * Y)
    clean = re.sub(
        r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)",
        r"((\1 / 100.0) * \2)",
        clean,
        flags=re.IGNORECASE
    )

    # Pattern: X% -> (X / 100.0)
    clean = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1 / 100.0)", clean)

    # Convert ^ to ** for power
    clean = clean.replace("^", "**")

    return clean.strip().rstrip("?.!;, ")


def _eval_node(node: ast.AST) -> Union[int, float]:
    """Recursively evaluate an AST node adhering to strict whitelist rules."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    # Numbers and constants
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise CalculatorSecurityError(f"Unsupported constant type: {type(node.value)}")

    # Legacy Num support
    elif hasattr(ast, "Num") and isinstance(node, ast.Num):
        return node.n

    # Constants like pi, e
    elif isinstance(node, ast.Name):
        if node.id in SAFE_FUNCTIONS and isinstance(SAFE_FUNCTIONS[node.id], (int, float)):
            return SAFE_FUNCTIONS[node.id]
        raise CalculatorSecurityError(f"Variable or function '{node.id}' is not an accessible variable.")

    # Unary operations (e.g. -5, +3)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            operand_val = _eval_node(node.operand)
            return SAFE_OPERATORS[op_type](operand_val)
        raise CalculatorSecurityError(f"Unsupported unary operator: {op_type}")

    # Binary operations (e.g. 5 + 3, 2 ** 8)
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            left_val = _eval_node(node.left)
            right_val = _eval_node(node.right)

            # Prevent excessive computation (e.g. 9999 ** 99999)
            if op_type == ast.Pow:
                if abs(left_val) > 1e4 or abs(right_val) > 1e4:
                    raise ValueError("Exponent or base too large for safe computation.")

            return SAFE_OPERATORS[op_type](left_val, right_val)
        raise CalculatorSecurityError(f"Unsupported binary operator: {op_type}")

    # Function calls (e.g. sqrt(16), round(3.1415, 2))
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise CalculatorSecurityError("Dynamic function dispatch is prohibited.")

        func_name = node.func.id
        if func_name not in SAFE_FUNCTIONS or not callable(SAFE_FUNCTIONS[func_name]):
            raise CalculatorSecurityError(f"Function '{func_name}' is not permitted.")

        func = SAFE_FUNCTIONS[func_name]
        args = [_eval_node(arg) for arg in node.args]

        try:
            return func(*args)
        except Exception as e:
            raise ValueError(f"Error evaluating function '{func_name}': {str(e)}")

    else:
        raise CalculatorSecurityError(
            f"Prohibited syntax detected: {type(node).__name__}. Only basic arithmetic is permitted."
        )


def safe_calculate(expression: str) -> str:
    """
    Safely evaluate an arithmetic expression using AST parsing.

    Args:
        expression: String mathematical query (e.g. '25% of 480', 'sqrt(144) + 10 * 2').

    Returns:
        String result or error description.
    """
    if not expression or not expression.strip():
        return "Error: Expression cannot be empty."

    try:
        normalized = preprocess_expression(expression)
        parsed = ast.parse(normalized, mode="eval")
        result = _eval_node(parsed)

        # Format integer results cleanly without trailing .0
        if isinstance(result, float) and result.is_integer():
            return f"Result: {int(result)}"
        elif isinstance(result, float):
            return f"Result: {round(result, 6)}"
        return f"Result: {result}"

    except ZeroDivisionError:
        return "Error: Division by zero."
    except CalculatorSecurityError as cse:
        return f"Security Error: {str(cse)}"
    except ValueError as ve:
        return f"Calculation Error: {str(ve)}"
    except SyntaxError:
        return f"Syntax Error: Could not parse mathematical expression '{expression}'."
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"
