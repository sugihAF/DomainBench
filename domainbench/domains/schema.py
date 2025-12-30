"""
Domain schema definitions
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class PersonaSchema(BaseModel):
    """Schema for a persona/character in test scenarios"""
    id: str
    name: str
    traits: List[str] = Field(default_factory=list)
    description: Optional[str] = None


class TestScenarioSchema(BaseModel):
    """Schema for a test scenario category"""
    id: str
    category: str
    difficulty: str = "medium"
    templates: List[str] = Field(default_factory=list)
    variables: Dict[str, List[str]] = Field(default_factory=dict)


class EvaluationCriterionSchema(BaseModel):
    """Schema for evaluation criteria"""
    metric: str
    weight: float = 1.0
    description: Optional[str] = None


class FunctionSchema(BaseModel):
    """Schema for function definitions (for function calling tests)"""
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class DomainSchema(BaseModel):
    """
    Complete schema for a domain definition.
    
    Domains define the context, personas, scenarios, and evaluation
    criteria for benchmarking LLMs in a specific use case.
    """
    name: str
    description: Optional[str] = None
    version: str = "1.0"
    
    # Context and prompts
    system_prompt: str
    
    # Optional components
    personas: List[PersonaSchema] = Field(default_factory=list)
    test_scenarios: List[TestScenarioSchema] = Field(default_factory=list)
    evaluation_criteria: List[EvaluationCriterionSchema] = Field(default_factory=list)
    functions: List[FunctionSchema] = Field(default_factory=list)
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, path: str) -> "DomainSchema":
        """Load domain schema from YAML file"""
        import yaml
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Handle nested 'domain' key if present
        if 'domain' in data:
            data = data['domain']
        
        return cls(**data)
    
    def to_yaml(self, path: str) -> None:
        """Save domain schema to YAML file"""
        import yaml
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump({"domain": self.model_dump()}, f, default_flow_style=False)
    
    def to_domain_config(self):
        """Convert to DomainConfig for use in benchmark"""
        from domainbench.core.config import DomainConfig
        return DomainConfig(
            name=self.name,
            description=self.description,
            system_prompt=self.system_prompt,
            version=self.version,
            personas=[p.model_dump() for p in self.personas],
            evaluation_criteria=[e.model_dump() for e in self.evaluation_criteria],
            functions=[f.model_dump() for f in self.functions],
        )
