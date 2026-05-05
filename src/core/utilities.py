# src/core/utilities.py
import re
from typing import Dict, List

class TokenParser:
    """
    Utility: Deterministic extraction of structured tokens from LLM output.
    Allows the UI to render structured cards (opportunities, flags) from raw text.
    """
    @staticmethod
    def parse_opportunities(text: str) -> List[Dict[str, str]]:
        # Matches formats like [OPPORTUNITY:Tax Harvesting:0.92]
        pattern = r"\[OPPORTUNITY:(.*?):(.*?)\]"
        matches = re.findall(pattern, text)
        return [{"category": m[0].strip(), "confidence": m[1].strip()} for m in matches]

    @staticmethod
    def parse_compliance_flags(text: str) -> List[str]:
        # Matches formats like [FLAG:Compliance:Missing Signature]
        pattern = r"\[FLAG:Compliance:(.*?)\]"
        return [match.strip() for match in re.findall(pattern, text)]

class PromptBuilder:
    """
    Utility: Central service for assembling system prompts based on mode/role.
    Allows dynamic prompt shaping before handing off to the LangGraph agents.
    """
    @staticmethod
    def build_system_context(advisor_name: str, mode: str = "Standard") -> str:
        base_prompt = f"You are assisting {advisor_name}, a Wealth Advisor at AEON."
        
        if mode == "Executive Summary":
            return base_prompt + " Provide highly condensed, bulleted insights. No conversational filler."
        elif mode == "Client Meeting Prep":
            return base_prompt + " Focus purely on identifying agenda items, unresolved tasks, and immediate talking points."
        
        return base_prompt