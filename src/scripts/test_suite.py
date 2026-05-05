# src/scripts/test_suite.py
import os
import json
import sys
import uuid
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.workflows.wf_002 import build_WF_002_graph
from src.utils.prompt_manager import prompt_manager

def run_automated_tests():
    print("🧪 Starting Automated Domain Test Suite...")
    
    # 1. Load the Target Questions (Our Unit Tests)
    base_dir = os.path.dirname(__file__)
    wf_path = os.path.abspath(os.path.join(base_dir, '../../workflows/workflow_2_portfolio_performance.json'))
    with open(wf_path, 'r') as f:
        wf_data = json.load(f)
    
    test_questions = wf_data.get("target_questions", [])
    app = build_WF_002_graph()
    
    # 2. Initialize the Critic for Evaluation
    critic_llm = ChatBedrock(
        model_id="us.anthropic.claude-sonnet-4-6",
        region_name="us-east-1",
        temperature=0.0
    )
    
    results = []

    for test in test_questions:
        q_id = test["id"]
        query = test["prompt"]
        print(f"\n▶️ Testing {q_id}: {test['intent']}...")
        
        # Execute the Graph
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        final_state = app.invoke(
            {"messages": [HumanMessage(content=query)]}, 
            config=config
        )
        
        answer = final_state["messages"][-1].content
        
        # 3. Critic Evaluation
        # 3. Critic Evaluation
        # 🛑 Use the Prompt Manager instead of hardcoding
        eval_prompt = prompt_manager.get_prompt("critic_eval_prompt", query=query, answer=answer)
        
        eval_result = critic_llm.invoke(eval_prompt).content
        
        status = "✅ PASS" if "PASS" in eval_result.upper() else "❌ FAIL"
        print(f"{status} - {eval_result}")
        results.append({"id": q_id, "status": status, "feedback": eval_result})

    # 4. Final Summary
    print("\n" + "="*30)
    print("📊 TEST SUITE SUMMARY")
    print("="*30)
    for r in results:
        print(f"{r['id']}: {r['status']}")

if __name__ == "__main__":
    run_automated_tests()