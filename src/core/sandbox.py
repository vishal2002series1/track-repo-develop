# src/core/sandbox.py
import os
import uuid
import importlib
from langchain_core.messages import HumanMessage

import src.agents.graph
from src.agents.graph import build_aeon_graph
from src.core.evaluator import GroundingCritic
from src.agents.fabricator import AgentFabricator
from src.agents.config import AgentConfig, registry_manager

# 🛑 CRITICAL: Suppress aggressive C++ gRPC logging from OpenTelemetry
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

class SandboxOrchestrator:
    def __init__(self):
        self.critic = GroundingCritic()
        self.fabricator = AgentFabricator()

    def extract_tool_logs(self, messages):
        tool_logs = []
        for m in messages:
            if getattr(m, "type", None) == "tool":
                content = str(m.content)
                # 🛑 FIX: Intelligently truncate massive SQL returns per tool call
                # 3000 chars is enough to show the headers and first ~15 rows to the Critic
                if len(content) > 3000:
                    content = content[:3000] + f"\n\n... [DATA TRUNCATED FOR CRITIC. Original length: {len(content)} chars. The Agent saw the full data.]"
                
                tool_logs.append(f"[Tool: {m.name}]: {content}")
        return "\n".join(tool_logs) if tool_logs else "No tools were called."

    # 🛑 FIX: Real-time active node tracking added here
    def execute_with_live_logs(self, app, input_messages, config):
        """Streams LangGraph events to the terminal so we aren't staring at a blank screen."""
        print("\n   [Live Execution Trace Started] ---------------------------")
        
        active_nodes = set()
        
        # Stream updates as they happen
        for event in app.stream(input_messages, config=config, stream_mode="updates"):
            for node_name, node_data in event.items():
                active_nodes.add(node_name)
                print(f"   🟢 [ACTIVE AGENT]: '{node_name}' is thinking...")
                
                # Did the supervisor make a routing decision?
                if "routing_log" in node_data and node_data["routing_log"]:
                    print(f"      🔀 Router decided: {node_data['routing_log'][-1]}")
                
                # Did an agent call a tool or get tool results?
                if "messages" in node_data and node_data["messages"]:
                    last_msg = node_data["messages"][-1]
                    
                    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                        for tc in last_msg.tool_calls:
                            print(f"      🛠️  Executing Tool: `{tc['name']}`...")
                    
                    elif getattr(last_msg, 'type', '') == 'tool':
                        print(f"      ✅ Tool returned {len(str(last_msg.content))} characters of data.")
        
        print("   [Live Execution Trace Complete] --------------------------\n")
        
        # Safely inject our tracked nodes into the final state payload
        state = app.get_state(config).values
        state["_tracked_active_nodes"] = list(active_nodes)
        return state

    def run_workflow_builder(self, test_query: str, max_iterations: int = 3):
        print("="*60)
        print(f"🏗️  AGENT BUILDER SANDBOX INITIATED")
        print(f"🎯 Goal: {test_query}")
        print("="*60)

        print("\n▶️ PHASE 1: Testing Existing Agent Registry...")
        app = build_aeon_graph()
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        
        state = self.execute_with_live_logs(app, {"messages": [HumanMessage(content=test_query)]}, config)
        
        final_answer = state["messages"][-1].content
        tool_logs = self.extract_tool_logs(state["messages"])
        
        print(f"\n[Existing Agents Answer]:\n{final_answer[:500]}... (truncated for brevity)\n")
        
        eval_result = self.critic.evaluate(test_query, tool_logs, final_answer)
        
        missing_phrases = ["unable to", "cannot find", "do not have", "no data was returned", "no relevant", "ambiguous_entity", "ask_human"]
        is_missing_capability = any(phrase in final_answer.lower() for phrase in missing_phrases)

        if eval_result.status == "PASS" and not is_missing_capability:
            print("\n   Testing existing agents via Sandbox Supervisor (Negative Validation Run)...")
            neg_query = f"NEGATIVE TEST: Execute the following query but explicitly target a non-existent entity named 'FAKE_ENTITY_999': {test_query}"
            neg_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            
            neg_state = self.execute_with_live_logs(app, {"messages": [HumanMessage(content=neg_query)]}, neg_config)
            neg_answer = neg_state["messages"][-1].content
            
            if any(kw in neg_answer.lower() for kw in ["ask_human", "ambiguous", "not found", "cannot", "unable", "fake_entity_999"]):
                print("✅ SUCCESS: Existing agents handled BOTH positive and negative tests perfectly.")
                
                # 🛑 FIX: Use the reliable tracker from execute_with_live_logs
                used_agents = list(set([a for a in state.get("_tracked_active_nodes", []) if a not in ["supervisor", "__interrupt__"]]))
                print(f"🗺️  Asking Fabricator to map the DAG for verified agents: {used_agents}...")
                
                feedback = f"SUCCESS: The query was completely solved and negatively validated using ONLY these existing agents: {used_agents}. Output the DAG edges mapping how these specific agents should be wired together (START -> workers -> synthesis -> END). DO NOT create any new agents."
                fab_out = self.fabricator.fabricate(test_query, critic_feedback=feedback)
                
                # Guaranteed to return a list of STRINGS
                return used_agents, fab_out.edges
            else:
                print("❌ PHASE 1 FAILED: Existing agents passed the positive test, but hallucinated on the negative test!")
                eval_result.status = "FAIL"
                eval_result.reason = "Failed Negative Test. Existing agents hallucinated a fuzzy match for a non-existent entity instead of halting."

        print("❌ GAP DETECTED: Existing agents failed, lacked capabilities, or failed negative validation.")
        if eval_result.status == "FAIL":
            print(f"Reason (Hallucination): {eval_result.reason}")
        else:
            print("Reason (Missing Capability): The agent correctly admitted it lacked the data/tools to answer fully.")

        print("\n▶️ PHASE 2: Fabricating New Agent(s) to Fill Gap...")
        feedback = eval_result.reason if eval_result.status == "FAIL" else "The previous agents lacked the tools or focus to find the data. Build specialized atomic agents with the correct tools."
        
        for iteration in range(1, max_iterations + 1):
            print(f"\n🔄 Fabrication Loop {iteration}/{max_iterations}")
            
            registry_backup = dict(registry_manager.agents)
            
            fabricator_output = self.fabricator.fabricate(test_query, critic_feedback=feedback)
            
            print(f"   Drafted {len(fabricator_output.new_agents)} granular agent(s):")
            temp_configs = []
            
            for bp in fabricator_output.new_agents:
                print(f"     - {bp.name} (Tools: {bp.authorized_tools})")
                temp_config = AgentConfig(
                    name=bp.name,
                    routing_description=bp.routing_description,
                    persona=bp.persona,
                    authorized_tools=bp.authorized_tools
                )
                temp_configs.append(temp_config)
                registry_manager.agents[temp_config.name] = temp_config

            importlib.reload(src.agents.graph)
            temp_app = src.agents.graph.build_aeon_graph()
            
            print("   Testing drafted agents via Sandbox Supervisor (Positive Test)...")
            sandbox_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            sandbox_state = self.execute_with_live_logs(temp_app, {"messages": [HumanMessage(content=test_query)]}, sandbox_config)
            
            sandbox_answer = sandbox_state["messages"][-1].content
            sandbox_tool_logs = self.extract_tool_logs(sandbox_state["messages"])

            print("   Testing drafted agents via Sandbox Supervisor (Negative Validation Run)...")
            neg_query = f"NEGATIVE TEST: Execute the following query but explicitly target a non-existent entity named 'FAKE_ENTITY_999': {test_query}"
            neg_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            neg_state = self.execute_with_live_logs(temp_app, {"messages": [HumanMessage(content=neg_query)]}, neg_config)
            neg_answer = neg_state["messages"][-1].content
            
            new_eval = self.critic.evaluate(test_query, sandbox_tool_logs, sandbox_answer)
            
            if not any(kw in neg_answer.lower() for kw in ["ask_human", "ambiguous", "not found", "cannot", "unable", "fake_entity_999"]):
                new_eval.status = "FAIL"
                new_eval.reason = f"Failed Negative Test. Agent hallucinated a fuzzy match for a non-existent entity instead of halting."
            
            if new_eval.status == "PASS":
                print(f"\n✅ CRITIC APPROVED! Version {iteration} passed strict grounding and negative fallback validation.")
                for tc in temp_configs:
                    registry_manager.save_agent(tc) 
                print(f"🎉 {len(temp_configs)} new granular agent(s) are now permanently available to the Supervisor!")
                
                # Guaranteed to return a list of STRINGS
                return [tc.name for tc in temp_configs], fabricator_output.edges
            else:
                print(f"   ⚠️ Critic Failed Draft {iteration}: {new_eval.reason}")
                feedback = new_eval.reason 
                registry_manager.agents = registry_backup

        print("\n🚨 MAX ITERATIONS REACHED. The Fabricator could not build a passing agent.")
        return [], []