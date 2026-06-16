"""Output parser with JSON extraction and repair heuristics.

Provides robust parsing of model outputs that may contain JSON embedded
in arbitrary text, code fences, or surrounding prose. Applies repair
heuristics when direct parsing fails.
"""

import json
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParseResult:
    """Result of parsing a model output string.

    Attributes:
        success: Whether JSON was successfully extracted and parsed.
        parsed_output: The parsed JSON object/array, or None on failure.
        repair_applied: Whether a repair heuristic was used.
        repair_type: Type of repair applied, or None.
            One of: "bracket_completion", "trailing_comma", "quote_fix", None
        raw_output: The original input string.
    """

    success: bool
    parsed_output: Optional[dict]
    repair_applied: bool
    repair_type: Optional[str]
    raw_output: str


class OutputParser:
    """Extracts and parses JSON from model output text.

    Strategy:
    1. If raw_output is empty/whitespace-only -> immediate failure
    2. Try parsing the entire string as JSON first
    3. Look for JSON within code fences (```json ... ```)
    4. Find the first { or [ and try to extract from there to matching close
    5. If direct parsing fails, try repair heuristics in order:
       a. Complete unclosed brackets/braces
       b. Remove trailing commas before } or ]
       c. Replace single quotes with double quotes
    6. After each repair attempt, try parsing again
    """

    def parse(self, raw_output: str) -> ParseResult:
        """Main entry point for parsing model output.

        Args:
            raw_output: The raw text output from a model.

        Returns:
            ParseResult with success status and parsed data.
        """
        # Reject whitespace-only strings immediately
        if not raw_output or raw_output.isspace():
            return ParseResult(
                success=False,
                parsed_output=None,
                repair_applied=False,
                repair_type=None,
                raw_output=raw_output,
            )

        # Try direct extraction (no repair)
        extracted = self._extract_json(raw_output)
        if extracted is not None:
            try:
                parsed = json.loads(extracted)
                return ParseResult(
                    success=True,
                    parsed_output=parsed,
                    repair_applied=False,
                    repair_type=None,
                    raw_output=raw_output,
                )
            except (json.JSONDecodeError, ValueError):
                pass

        # Determine the candidate text for repair
        candidate = extracted if extracted is not None else raw_output

        # Try repair heuristics in order
        repair_attempts = [
            ("bracket_completion", self._complete_brackets),
            ("trailing_comma", self._remove_trailing_commas),
            ("quote_fix", self._fix_quotes),
        ]

        for repair_type, repair_fn in repair_attempts:
            repaired = repair_fn(candidate)
            if repaired != candidate:
                try:
                    parsed = json.loads(repaired)
                    return ParseResult(
                        success=True,
                        parsed_output=parsed,
                        repair_applied=True,
                        repair_type=repair_type,
                        raw_output=raw_output,
                    )
                except (json.JSONDecodeError, ValueError):
                    pass

        # All repair attempts failed
        return ParseResult(
            success=False,
            parsed_output=None,
            repair_applied=False,
            repair_type=None,
            raw_output=raw_output,
        )

    def _extract_json(self, text: str) -> Optional[str]:
        """Find the first JSON object or array embedded in arbitrary text.

        Tries multiple strategies:
        1. Parse the entire string as JSON
        2. Extract from code fences (```json ... ``` or ``` ... ```)
        3. Find first { or [ and extract to matching close bracket

        Args:
            text: The text that may contain embedded JSON.

        Returns:
            The extracted JSON string, or None if no JSON found.
        """
        # Strategy 1: Try the entire string as-is
        stripped = text.strip()
        if self._is_valid_json(stripped):
            return stripped

        # Strategy 2: Look for JSON in code fences
        code_fence_pattern = re.compile(
            r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL
        )
        match = code_fence_pattern.search(text)
        if match:
            fenced_content = match.group(1).strip()
            if fenced_content:
                return fenced_content

        # Strategy 3: Find first { or [ and extract to matching close
        return self._find_json_substring(text)

    def _find_json_substring(self, text: str) -> Optional[str]:
        """Find the first JSON object or array by bracket matching.

        Args:
            text: Text to search for JSON.

        Returns:
            The extracted substring, or None if no match found.
        """
        # Find first { or [
        first_brace = text.find("{")
        first_bracket = text.find("[")

        if first_brace == -1 and first_bracket == -1:
            return None

        # Pick whichever comes first
        if first_brace == -1:
            start = first_bracket
            open_char, close_char = "[", "]"
        elif first_bracket == -1:
            start = first_brace
            open_char, close_char = "{", "}"
        elif first_brace < first_bracket:
            start = first_brace
            open_char, close_char = "{", "}"
        else:
            start = first_bracket
            open_char, close_char = "[", "]"

        # Track bracket depth to find matching close
        depth = 0
        in_string = False
        escape_next = False

        for i in range(start, len(text)):
            ch = text[i]

            if escape_next:
                escape_next = False
                continue

            if ch == "\\":
                if in_string:
                    escape_next = True
                continue

            if ch == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if ch == open_char:
                depth += 1
            elif ch == close_char:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

        # If we didn't find a matching close, return from start to end
        # (this gives repair heuristics a chance to fix it)
        return text[start:]

    def _repair_json(self, text: str) -> Optional[str]:
        """Apply repair heuristics to make text parseable as JSON.

        Tries repairs in order:
        1. Complete unclosed brackets/braces
        2. Remove trailing commas
        3. Fix single quotes to double quotes

        Args:
            text: Potentially malformed JSON text.

        Returns:
            Repaired JSON string, or None if all repairs fail.
        """
        repairs = [
            self._complete_brackets,
            self._remove_trailing_commas,
            self._fix_quotes,
        ]

        for repair_fn in repairs:
            repaired = repair_fn(text)
            if self._is_valid_json(repaired):
                return repaired

        return None

    def _complete_brackets(self, text: str) -> str:
        """Complete unclosed brackets and braces.

        Counts open/close brackets and appends missing closing
        characters in the correct order.

        Args:
            text: JSON text with potentially missing closing brackets.

        Returns:
            Text with closing brackets appended as needed.
        """
        stack = []
        in_string = False
        escape_next = False

        for ch in text:
            if escape_next:
                escape_next = False
                continue

            if ch == "\\" and in_string:
                escape_next = True
                continue

            if ch == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if ch in ("{", "["):
                stack.append(ch)
            elif ch == "}":
                if stack and stack[-1] == "{":
                    stack.pop()
            elif ch == "]":
                if stack and stack[-1] == "[":
                    stack.pop()

        # Append missing closing brackets in reverse order
        closing = ""
        for opener in reversed(stack):
            if opener == "{":
                closing += "}"
            else:
                closing += "]"

        return text + closing

    def _remove_trailing_commas(self, text: str) -> str:
        """Remove trailing commas before } or ].

        Handles patterns like:
        - {"key": "value",}
        - [1, 2, 3,]
        - {"key": "value", }

        Args:
            text: JSON text with potential trailing commas.

        Returns:
            Text with trailing commas removed.
        """
        # Remove commas followed by optional whitespace and } or ]
        result = re.sub(r",\s*([}\]])", r"\1", text)
        return result

    def _fix_quotes(self, text: str) -> str:
        """Replace single quotes with double quotes for JSON compliance.

        Handles the common case where model output uses single quotes
        instead of double quotes for JSON strings.

        Args:
            text: JSON text with potential single quotes.

        Returns:
            Text with single quotes replaced by double quotes.
        """
        # Simple approach: replace single quotes that appear to be JSON delimiters
        # We need to be careful not to replace apostrophes within double-quoted strings
        result = []
        in_double_string = False
        escape_next = False

        for i, ch in enumerate(text):
            if escape_next:
                result.append(ch)
                escape_next = False
                continue

            if ch == "\\":
                result.append(ch)
                escape_next = True
                continue

            if ch == '"' and not escape_next:
                in_double_string = not in_double_string
                result.append(ch)
                continue

            if ch == "'" and not in_double_string:
                result.append('"')
            else:
                result.append(ch)

        return "".join(result)

    @staticmethod
    def _is_valid_json(text: str) -> bool:
        """Check if text is valid JSON.

        Args:
            text: String to check.

        Returns:
            True if text parses as valid JSON.
        """
        try:
            json.loads(text)
            return True
        except (json.JSONDecodeError, ValueError, TypeError):
            return False
