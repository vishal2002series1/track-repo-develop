# test_compiled_workflow.py
import os
import sys
import contextlib

# 1. Aggressively set environment variables BEFORE any other imports
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["GRPC_VERBOSITY"] = "NONE" # Changed from ERROR to NONE
os.environ["GRPC_TRACE"] = ""
os.environ["CHROMA_TELEMETRY_IMPL"] = "None" # Silence ChromaDB telemetry
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Silence HuggingFace warnings

import time
import uuid
import json
from langchain_core.messages import HumanMessage

# Import the new WF_001 graph!
from src.workflows.wf_001 import build_WF_001_graph

@contextlib.contextmanager
def suppress_cpp_logs():
    """Temporarily redirects stderr to /dev/null to silence C++ gRPC spam."""
    with open(os.devnull, "w") as devnull:
        old_stderr = os.dup(sys.stderr.fileno())
        os.dup2(devnull.fileno(), sys.stderr.fileno())
        try:
            yield
        finally:
            os.dup2(old_stderr, sys.stderr.fileno())
            os.close(old_stderr)

def run_performance_test():
    print("🚀 Loading Compiled Workflow (WF_001)...")
    
    # Wrap graph building in the suppressor just in case imports trigger logs
    with suppress_cpp_logs():
        app = build_WF_001_graph()
    
    base_dir = os.path.dirname(__file__)
    workflow_path = os.path.abspath(os.path.join(base_dir, 'workflows/workflow_1_client_prep.json'))
    
    with open(workflow_path, 'r') as f:
        workflow = json.load(f)
        
    queries = [q["prompt"] for q in workflow["target_questions"]]
    
    for i, query in enumerate(queries, 1):
        print("\n" + "═"*80)
        print(f"🎯 EXECUTING QUESTION {i}/{len(queries)}")
        print(f"❓ {query}")
        print("═"*80)
        
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        print("⏳ Running static graph...")
        start_time = time.time()
        
        final_answer = ""
        
        # 2. Wrap the stream execution in the C++ log suppressor
        with suppress_cpp_logs():
            for event in app.stream({"messages": [HumanMessage(content=query)]}, config=config, stream_mode="updates"):
                for node_name, node_data in event.items():
                    # We still want to print Python-level node completions, so we use sys.__stdout__
                    sys.__stdout__.write(f"  🟢 [NODE COMPLETED]: '{node_name}' finished its task.\n")
                    sys.__stdout__.flush()
                    if "messages" in node_data and len(node_data["messages"]) > 0:
                        final_answer = node_data["messages"][-1].content
                    
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"\n⏱️ EXECUTION TIME: {execution_time:.2f} seconds")
        print("\n📝 [FINAL ANSWER]:")
        print(final_answer)
        print("-" * 80)
        
        time.sleep(2)

if __name__ == "__main__":
    run_performance_test()