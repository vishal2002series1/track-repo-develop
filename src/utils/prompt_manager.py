# src/utils/prompt_manager.py
import json
import os

class PromptManager:
    def __init__(self):
        # 🟢 MOVED: Path points to the new Git-tracked config directory
        self.library_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/prompt_library.json'))
        self.prompts = self._load_library()

    def _load_library(self):
        if os.path.exists(self.library_path):
            with open(self.library_path, 'r') as f:
                return json.load(f)
        else:
            print(f"⚠️ Warning: Prompt library not found at {self.library_path}")
            return {}

    def get_prompt(self, prompt_id: str, **kwargs) -> str:
        """Fetches a prompt by ID and injects any required variables."""
        prompt = self.prompts.get(prompt_id)
        if not prompt:
            return f"[PROMPT NOT FOUND: {prompt_id}]"
            
        if kwargs:
            # Inject variables like {agent_names} or {query} into the prompt
            try:
                return prompt.format(**kwargs)
            except KeyError as e:
                print(f"⚠️ Warning: Missing formatting key {e} for prompt {prompt_id}")
                return prompt 
                
        return prompt

# Export a single instance to be used across the app
prompt_manager = PromptManager()