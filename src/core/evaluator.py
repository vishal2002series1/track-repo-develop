# src/core/evaluator.py
import os
from pydantic import BaseModel, Field
from typing import Literal

# 🟢 AZURE MIGRATION
from langchain_openai import AzureChatOpenAI

from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

class EvaluationResult(BaseModel):
    reason: str = Field(
        ..., 
        description="STEP 1: A detailed explanation of the attribution check. Note: The tool logs provided to you have been truncated to save space. If the agent's answer relies on data that logically belongs to a successfully executed query shown in the logs, give it the benefit of the doubt."
    )
    status: Literal["PASS", "FAIL"] = Field(
        ..., 
        description="STEP 2: PASS if the answer is grounded. FAIL if there is blatant hallucination of tools or errors."
    )

class GroundingCritic:
    def __init__(self): 
        # 🟢 AZURE MIGRATION: Dynamically pull credentials
        api_key = os.getenv("API_KEYS")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("OPENAI_API_VERSION")
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5.4")
        
        self.llm = AzureChatOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
            azure_deployment=deployment_name,
            temperature=0.0,
            max_tokens=2000
        ).with_structured_output(EvaluationResult)

        self.static_system_prompt = """You are an uncompromising Compliance Auditor for AEON Wealth Management.
Your exclusive job is to perform a "3-Way Match" to verify AI Agent Attribution.

THE RULES OF EVALUATION:
1. You will receive: The User's Question, the Raw Tool Logs, and the Agent's Final Answer.
2. ⚠️ IMPORTANT: The Tool Logs have been truncated for length. You will see the column headers and first few rows of SQL data. If the Agent cites a client name that fits the query executed, ASSUME THE DATA WAS IN THE FULL LOG and PASS the agent.
3. Admitting a lack of information is safe and should PASS.
4. NEGATIVE COMPLIANCE: If the agent encounters a fake entity (e.g., FAKE_ENTITY_999), it MUST halt or state it is missing.
5. CRITICAL: You MUST use the provided JSON schema to return your result. Never return empty."""

    def evaluate(self, user_question: str, tool_logs: str, agent_answer: str) -> EvaluationResult:
        dynamic_content = f"""Please perform the 3-Way Match Evaluation on the following:

--- 1. USER QUESTION ---
{user_question}

--- 2. RAW TOOL LOGS ---
{tool_logs}

--- 3. AGENT'S FINAL ANSWER ---
{agent_answer}
"""
        messages = [
            SystemMessage(content=self.static_system_prompt),
            HumanMessage(content=dynamic_content)
        ]
        
        try:
            result = self.llm.invoke(messages)
            if not result or not getattr(result, 'status', None):
                raise ValueError("Model returned empty or invalid JSON")
            return result
        except Exception as e:
            print(f"\n      ⚠️ [CRITIC WARNING]: Evaluator LLM failed to format JSON ({str(e)}). Defaulting to FAIL.")
            return EvaluationResult(
                reason="The Critic LLM experienced a structured output failure. Defaulting to FAIL.",
                status="FAIL"
            )