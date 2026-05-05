# chat_wf_001.py
import os
import uuid
import sys
import contextlib

# Suppress C++ gRPC spam
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GRPC_TRACE"] = ""
os.environ["CHROMA_TELEMETRY_IMPL"] = "None"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_core.messages import HumanMessage
from src.workflows.wf_001 import build_WF_001_graph

@contextlib.contextmanager
def suppress_cpp_logs():
    with open(os.devnull, "w") as devnull:
        old_stderr = os.dup(sys.stderr.fileno())
        os.dup2(devnull.fileno(), sys.stderr.fileno())
        try:
            yield
        finally:
            os.dup2(old_stderr, sys.stderr.fileno())
            os.close(old_stderr)

def chat():
    print("🚀 Loading Aeon Wealth Agent Factory (WF_001)...")
    with suppress_cpp_logs():
        app = build_WF_001_graph()
    
    # Generate a unique thread ID for this chat session
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    print("\n✅ System Ready. Type 'exit' to quit.\n")
    
    while True:
        user_input = input("🗣️ You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        if not user_input.strip():
            continue
            
        print("\n⏳ Thinking...")
        final_answer = ""
        
        with suppress_cpp_logs():
            for event in app.stream({"messages": [HumanMessage(content=user_input)]}, config=config, stream_mode="updates"):
                for node_name, node_data in event.items():
                    # Print progress so the user isn't staring at a blank screen
                    sys.__stdout__.write(f"  🟢 [AGENT ACTIVE]: '{node_name}' is processing...\n")
                    sys.__stdout__.flush()
                    if "messages" in node_data and len(node_data["messages"]) > 0:
                        final_answer = node_data["messages"][-1].content
        
        print("\n🤖 Aeon AI:")
        print(final_answer)
        print("\n" + "═"*80 + "\n")

if __name__ == "__main__":
    chat()