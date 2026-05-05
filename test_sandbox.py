# test_sandbox.py
from src.core.sandbox import SandboxOrchestrator

if __name__ == "__main__":
    orchestrator = SandboxOrchestrator()
    
    # We are asking for a highly specific query about a fake entity
    test_query = "Summarize the estate planning documents and recent meeting notes for the entirely fictional 'Skywalker Family'."
    
    orchestrator.run_workflow_builder(test_query)