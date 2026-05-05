# chat_wf_003.py
import uuid
from langchain_core.messages import HumanMessage
from src.workflows.wf_003 import build_WF_003_graph

# The AI's thought trace
thought_trace = []

def chat_loop():
    print("\n" + "="*50)
    print("🤖 AEON AI: Meeting Strategy & Next Best Action")
    print("="*50)
    print("(Type 'exit' or 'quit' to end the session)\n")

    # Generate a unique session ID for memory checkpointing
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    
    # Compile the graph once
    app = build_WF_003_graph()

    while True:
        try:
            user_input = input("\n🧑‍💼 You: ")
            if user_input.lower() in ['quit', 'exit']:
                print("\nEnding session. Goodbye!")
                break
            if not user_input.strip():
                continue

            print("\n⏳ AI is thinking...")
            thought_trace.clear()
            final_answer = ""

            # Stream the graph execution
            for event in app.stream({"messages": [HumanMessage(content=user_input)]}, config=config, stream_mode="updates"):
                if not event:
                    continue
                    
                for node_name, node_data in event.items():
                    print(f"   [Routing] -> 🟢 {node_name} is processing...")
                    thought_trace.append(node_name)
                    
                    if node_data is not None:
                        messages = node_data.get("messages")
                        if messages and isinstance(messages, list) and len(messages) > 0:
                            if hasattr(messages[-1], 'content') and messages[-1].content:
                                final_answer = messages[-1].content

            print("\n" + "="*50)
            print("🤖 Aeon AI:")
            print(final_answer)
            print("="*50)
            
            # Print the X-Ray trace for transparency
            print(f"🔍 Agent X-Ray Trace: {' -> '.join(thought_trace)}")

        except Exception as e:
            print(f"\n❌ Execution Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    chat_loop()