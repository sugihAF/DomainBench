"""
Base capability interface
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseCapability(ABC):
    """
    Abstract base class for benchmark capabilities.
    
    Each capability (chat, function calling, etc.) implements this interface.
    """
    
    name: str = "base"
    description: str = "Base capability"
    required_provider_features: List[str] = []
    
    @abstractmethod
    def build_messages(
        self,
        test_case: Dict[str, Any],
        system_prompt: str,
    ) -> List[Dict[str, str]]:
        """
        Build messages/prompt from a test case.
        
        Args:
            test_case: Test case data (e.g., with 'turns' for multi-turn)
            system_prompt: System prompt from domain config
            
        Returns:
            List of message dicts for the LLM API
        """
        pass
    
    @abstractmethod
    def validate_test_case(self, test_case: Dict[str, Any]) -> bool:
        """
        Validate that a test case has required fields for this capability.
        
        Args:
            test_case: Test case data
            
        Returns:
            True if valid, False otherwise
        """
        pass
    
    def get_metrics(self) -> List[str]:
        """Get list of metrics this capability collects"""
        return ["accuracy", "latency", "tokens"]
    
    def get_required_fields(self) -> List[str]:
        """Get list of required fields in test cases"""
        return ["id"]
