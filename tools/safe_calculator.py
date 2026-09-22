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


NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
    "million": 1000000, "billion": 1000000000,
}

UNITS_TEENS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}

TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

SCALES = {
    "hundred": 100, "thousand": 1000, "million": 1000000, "billion": 1000000000,
}


def words_to_number_sequence(phrase: str) -> str:
    """Convert English number words ('twenty five', 'one hundred') into numeric digits ('25', '100')."""
    words = phrase.replace("-", " ").split()
    tokens = []
    i = 0
    while i < len(words):
        w = words[i].lower()
        if w in NUMBER_WORDS or (w == "and" and i > 0 and i + 1 < len(words) and words[i+1].lower() in NUMBER_WORDS):
            num_words = []
            while i < len(words):
                curr = words[i].lower()
                if curr in NUMBER_WORDS:
                    num_words.append(curr)
                    i += 1
                elif curr == "and" and i + 1 < len(words) and words[i+1].lower() in NUMBER_WORDS:
                    i += 1
                else:
                    break

            total = 0
            current = 0
            for nw in num_words:
                if nw in UNITS_TEENS:
                    current += UNITS_TEENS[nw]
                elif nw in TENS:
                    current += TENS[nw]
                elif nw == "hundred":
                    current = max(1, current) * 100
                elif nw in ("thousand", "million", "billion"):
                    current = max(1, current) * SCALES[nw]
                    total += current
                    current = 0
            total += current
            tokens.append(str(total))
        else:
            tokens.append(words[i])
            i += 1
    return " ".join(tokens)


def preprocess_expression(expr: str) -> str:
    """
    Clean and normalize user mathematical queries.
    Handles number words ('two plus two'), idioms like '25% of 480', '2^10', etc.
    """
    clean = expr.strip().rstrip("?.!;, ")

    # Strip conversational prefixes
    clean = re.sub(
        r"^(?:what\s+is|calculate|compute|evaluate|solve|how\s+much\s+is|can\s+you\s+calculate|find)\s+",
        "",
        clean,
        flags=re.IGNORECASE
    ).strip()

    # Convert English number words to digits
    clean = words_to_number_sequence(clean)

    # Convert word operators to mathematical symbols
    clean = re.sub(r"\b(?:square\s+root\s+of|sqrt\s+of)\s+([0-9\.\(\)]+)", r"sqrt(\1)", clean, flags=re.IGNORECASE)
    clean = re.sub(r"([0-9\.\(\)]+)\s+squared\b", r"(\1 ** 2)", clean, flags=re.IGNORECASE)
    clean = re.sub(r"([0-9\.\(\)]+)\s+cubed\b", r"(\1 ** 3)", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\b(?:raised\s+to\s+(?:the\s+)?power\s+(?:of\s+)?|to\s+the\s+power\s+(?:of\s+)?|power)\b", r"**", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\b(?:multiplied\s+by|times|into)\b", r"*", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\b(?:divided\s+by|divide\s+by|over)\b", r"/", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\b(?:modulo|mod)\b", r"%", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\b(?:added\s+to|plus|add)\b", r"+", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\b(?:minus|subtract)\b", r"-", clean, flags=re.IGNORECASE)

    # Handle "X subtracted from Y" -> "Y - X"
    sub_from = re.search(r"([0-9\.\s\+\*\/\%\^\(\)]+)\s+subtracted\s+from\s+([0-9\.\s\+\*\/\%\^\(\)]+)", clean, flags=re.IGNORECASE)
    if sub_from:
        clean = f"({sub_from.group(2)}) - ({sub_from.group(1)})"

    # Pattern: X% of Y or X percent of Y -> ((X / 100) * Y)
    clean = re.sub(
        r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*of\s*(\d+(?:\.\d+)?)",
        r"((\1 / 100.0) * \2)",
        clean,
        flags=re.IGNORECASE
    )

    # Pattern: X% or X percent -> (X / 100.0)
    clean = re.sub(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", r"(\1 / 100.0)", clean, flags=re.IGNORECASE)

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
