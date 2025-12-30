"""
Configuration models for DomainBench using Pydantic
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class ProviderType(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    CUSTOM = "custom"


class ModelConfig(BaseModel):
    """Configuration for a single model to benchmark"""
    provider: ProviderType
    model: str
    alias: Optional[str] = None
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    api_key_env: Optional[str] = None  # Environment variable name for API key
    
    @property
    def display_name(self) -> str:
        return self.alias or f"{self.provider.value}/{self.model}"


class JudgeConfig(BaseModel):
    """Configuration for the LLM judge"""
    provider: ProviderType = ProviderType.OPENAI
    model: str = "gpt-4o"
    temperature: float = 0.0
    api_key_env: Optional[str] = None


class MetricsConfig(BaseModel):
    """Configuration for which metrics to collect"""
    accuracy: bool = True
    latency: bool = True
    cost: bool = True
    tokens: bool = True


class DomainConfig(BaseModel):
    """Configuration for a benchmark domain"""
    name: str
    description: Optional[str] = None
    system_prompt: str
    version: str = "1.0"
    
    # Optional domain-specific settings
    personas: List[Dict[str, Any]] = Field(default_factory=list)
    evaluation_criteria: List[Dict[str, Any]] = Field(default_factory=list)
    functions: List[Dict[str, Any]] = Field(default_factory=list)  # For function calling tests


class BenchmarkSettings(BaseModel):
    """Runtime settings for benchmark execution"""
    runs_per_test: int = 1
    parallel_execution: bool = False
    save_raw_responses: bool = True
    seed: Optional[int] = None
    sleep_between_calls: float = 0.2
    max_items: Optional[int] = None


class OutputConfig(BaseModel):
    """Configuration for benchmark output"""
    formats: List[str] = Field(default_factory=lambda: ["json"])
    directory: str = "./results"
    include_raw_responses: bool = True


class BenchmarkConfig(BaseModel):
    """Main benchmark configuration"""
    name: str
    description: Optional[str] = None
    
    models: List[ModelConfig]
    capabilities: List[str] = Field(default_factory=lambda: ["chat_completion"])
    
    domain: str  # Domain name or path
    domain_config: Optional[DomainConfig] = None  # Loaded domain config
    
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    settings: BenchmarkSettings = Field(default_factory=BenchmarkSettings)
    output: OutputConfig = Field(default_factory=OutputConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "BenchmarkConfig":
        """Load config from YAML file"""
        import yaml
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Handle nested 'benchmark' key if present
        if 'benchmark' in data:
            data = data['benchmark']
        
        return cls(**data)
    
    def to_yaml(self, path: str) -> None:
        """Save config to YAML file"""
        import yaml
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump({"benchmark": self.model_dump()}, f, default_flow_style=False)
