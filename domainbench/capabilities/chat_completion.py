"""
Chat Completion Capability - Multi-turn conversation benchmark
Based on the waiterbench.py conversation handling
"""

from typing import List, Dict, Any
from domainbench.capabilities.base import BaseCapability


class ChatCompletionCapability(BaseCapability):
    """
    Chat completion capability for multi-turn conversations.
    
    Test cases should have a 'turns' field with a list of user messages.
    The model responds to all turns at once (MT-Bench style).
    """
    
    name = "chat_completion"
    description = "Multi-turn chat conversation benchmark"
    required_provider_features = ["chat_completion"]
    
    def build_messages(
        self,
        test_case: Dict[str, Any],
        system_prompt: str,
    ) -> List[Dict[str, str]]:
        """
        Build messages from multi-turn test case.
        
        The conversation is formatted as:
        - System message with domain context
        - All user turns as separate user messages
        
        Args:
            test_case: Dict with 'turns' (list of user messages)
            system_prompt: System prompt from domain config
            
        Returns:
            List of messages for chat completion API
        """
        messages = []
        
        # Add system prompt
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })
        
        # Add all user turns
        turns = test_case.get("turns", [])
        for turn in turns:
            messages.append({
                "role": "user",
                "content": turn,
            })
        
        return messages
    
    def validate_test_case(self, test_case: Dict[str, Any]) -> bool:
        """Validate test case has required 'turns' field"""
        if "turns" not in test_case:
            return False
        
        turns = test_case["turns"]
        if not isinstance(turns, list) or len(turns) == 0:
            return False
        
        return True
    
    def get_required_fields(self) -> List[str]:
        """Required fields for chat completion test cases"""
        return ["id", "turns"]
    
    def format_conversation_for_display(self, test_case: Dict[str, Any]) -> str:
        """Format conversation turns for display/logging"""
        turns = test_case.get("turns", [])
        return "\n".join([f"USER[{i+1}]: {t}" for i, t in enumerate(turns)])
