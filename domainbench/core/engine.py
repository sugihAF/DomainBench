"""
Benchmark Engine - Main orchestrator for running benchmarks
"""

import json
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from domainbench.core.config import BenchmarkConfig, ModelConfig
from domainbench.core.evaluator import JudgeEvaluator
from domainbench.core.reporter import Reporter
from domainbench.providers import get_provider, BaseProvider
from domainbench.capabilities import get_capability, BaseCapability


class BenchmarkResult(Dict[str, Any]):
    """Container for a single test case result"""
    pass


class BenchmarkEngine:
    """
    Main orchestrator for running benchmarks.
    
    Coordinates between:
    - Providers (LLM APIs)
    - Capabilities (what to test)
    - Domains (context and test cases)
    - Evaluators (scoring)
    - Reporter (output)
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.benchmark_id = str(uuid.uuid4())
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # Initialize components
        self.providers: Dict[str, BaseProvider] = {}
        self.capabilities: Dict[str, BaseCapability] = {}
        self.evaluator: Optional[JudgeEvaluator] = None
        self.reporter = Reporter(config.output)
        
        # Results storage
        self.results: List[BenchmarkResult] = []
        self.summary: Dict[str, Any] = {}
    
    def setup(self) -> None:
        """Initialize all components before running"""
        # Import here to avoid circular import
        from domainbench.domains.loader import load_domain
        
        # Load domain config if not already loaded
        if self.config.domain_config is None:
            self.config.domain_config = load_domain(self.config.domain)
        
        # Initialize providers for each model
        for model_config in self.config.models:
            provider = get_provider(model_config)
            self.providers[model_config.display_name] = provider
        
        # Initialize capabilities
        for cap_name in self.config.capabilities:
            capability = get_capability(cap_name)
            self.capabilities[cap_name] = capability
        
        # Initialize evaluator (judge)
        judge_provider = get_provider(ModelConfig(
            provider=self.config.judge.provider,
            model=self.config.judge.model,
            temperature=self.config.judge.temperature,
            api_key_env=self.config.judge.api_key_env,
        ))
        self.evaluator = JudgeEvaluator(judge_provider, self.config.judge.model)
    
    def run(self, dataset_path: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Run the benchmark on a dataset.
        
        Args:
            dataset_path: Path to JSONL dataset file
            verbose: Print progress to console
            
        Returns:
            Complete benchmark results dictionary
        """
        from rich.console import Console
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
        from domainbench.domains.loader import load_dataset
        
        console = Console()
        
        self.setup()
        self.start_time = datetime.now()
        
        # Load dataset
        dataset = load_dataset(dataset_path)
        if self.config.settings.max_items:
            dataset = dataset[:self.config.settings.max_items]
        
        if verbose:
            console.print(f"\n[bold blue]DomainBench[/bold blue] - {self.config.name}")
            console.print(f"Running {len(dataset)} test cases across {len(self.config.models)} models\n")
        
        # Track wins/scores per model per capability
        model_stats: Dict[str, Dict[str, Dict[str, Any]]] = {
            model.display_name: {cap: {"wins": 0, "ties": 0, "losses": 0, "scores": []} 
                                  for cap in self.config.capabilities}
            for model in self.config.models
        }
        
        # Category-level tracking
        category_stats: Dict[str, Dict[str, Dict[str, int]]] = {}
        
        # We need exactly 2 models for comparison
        if len(self.config.models) != 2:
            raise ValueError("Currently only 2-model comparison is supported")
        
        model_a = self.config.models[0]
        model_b = self.config.models[1]
        provider_a = self.providers[model_a.display_name]
        provider_b = self.providers[model_b.display_name]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
            disable=not verbose,
        ) as progress:
            task = progress.add_task("Running benchmark...", total=len(dataset))
            
            for idx, test_case in enumerate(dataset):
                case_id = test_case.get("id", f"case_{idx}")
                category = test_case.get("category", "unknown")
                
                if category not in category_stats:
                    category_stats[category] = {
                        model_a.display_name: {"wins": 0, "ties": 0},
                        model_b.display_name: {"wins": 0, "ties": 0},
                    }
                
                for cap_name, capability in self.capabilities.items():
                    # Generate responses from both models
                    response_a, latency_a, tokens_a = self._generate_response(
                        provider_a, model_a, capability, test_case
                    )
                    response_b, latency_b, tokens_b = self._generate_response(
                        provider_b, model_b, capability, test_case
                    )
                    
                    # Judge evaluation with swap mitigation
                    judge_result = self.evaluator.evaluate_pair(
                        conversation=test_case.get("turns", []),
                        response_a=response_a,
                        response_b=response_b,
                        system_prompt=self.config.domain_config.system_prompt,
                    )
                    
                    winner = judge_result["winner"]
                    
                    # Update stats
                    if winner == "A":
                        model_stats[model_a.display_name][cap_name]["wins"] += 1
                        model_stats[model_b.display_name][cap_name]["losses"] += 1
                        category_stats[category][model_a.display_name]["wins"] += 1
                    elif winner == "B":
                        model_stats[model_b.display_name][cap_name]["wins"] += 1
                        model_stats[model_a.display_name][cap_name]["losses"] += 1
                        category_stats[category][model_b.display_name]["wins"] += 1
                    else:
                        model_stats[model_a.display_name][cap_name]["ties"] += 1
                        model_stats[model_b.display_name][cap_name]["ties"] += 1
                        category_stats[category][model_a.display_name]["ties"] += 1
                        category_stats[category][model_b.display_name]["ties"] += 1
                    
                    model_stats[model_a.display_name][cap_name]["scores"].append(judge_result["score_A"])
                    model_stats[model_b.display_name][cap_name]["scores"].append(judge_result["score_B"])
                    
                    # Store detailed result
                    result = BenchmarkResult({
                        "test_id": case_id,
                        "category": category,
                        "capability": cap_name,
                        "input": test_case,
                        "responses": {
                            model_a.display_name: {
                                "response": response_a,
                                "latency_ms": latency_a,
                                "tokens": tokens_a,
                                "score": judge_result["score_A"],
                            },
                            model_b.display_name: {
                                "response": response_b,
                                "latency_ms": latency_b,
                                "tokens": tokens_b,
                                "score": judge_result["score_B"],
                            },
                        },
                        "winner": winner,
                        "judge_reasons": judge_result.get("reasons", []),
                    })
                    self.results.append(result)
                
                progress.update(task, advance=1)
                
                # Sleep between calls
                if self.config.settings.sleep_between_calls > 0:
                    time.sleep(self.config.settings.sleep_between_calls)
        
        self.end_time = datetime.now()
        
        # Build summary
        self.summary = self._build_summary(model_stats, category_stats, len(dataset))
        
        if verbose:
            self._print_summary(console)
        
        return self.get_full_results()
    
    def _generate_response(
        self,
        provider: BaseProvider,
        model_config: ModelConfig,
        capability: BaseCapability,
        test_case: Dict[str, Any],
    ) -> Tuple[str, float, int]:
        """Generate a response from a model and measure metrics"""
        
        # Build messages from test case
        messages = capability.build_messages(
            test_case=test_case,
            system_prompt=self.config.domain_config.system_prompt,
        )
        
        start = time.perf_counter()
        response = provider.chat_completion(
            model=model_config.model,
            messages=messages,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        
        # Extract token count if available
        tokens = response.get("usage", {}).get("total_tokens", 0)
        content = response.get("content", "")
        
        return content, latency_ms, tokens
    
    def _build_summary(
        self,
        model_stats: Dict[str, Dict[str, Dict[str, Any]]],
        category_stats: Dict[str, Dict[str, Dict[str, int]]],
        total_cases: int,
    ) -> Dict[str, Any]:
        """Build summary statistics"""
        
        summary = {
            "total_test_cases": total_cases,
            "models": {},
            "by_capability": {},
            "by_category": category_stats,
        }
        
        for model_name, cap_stats in model_stats.items():
            model_summary = {
                "total_wins": 0,
                "total_ties": 0,
                "total_losses": 0,
                "avg_score": 0.0,
                "by_capability": {},
            }
            
            all_scores = []
            for cap_name, stats in cap_stats.items():
                model_summary["total_wins"] += stats["wins"]
                model_summary["total_ties"] += stats["ties"]
                model_summary["total_losses"] += stats["losses"]
                all_scores.extend(stats["scores"])
                
                cap_avg = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0
                model_summary["by_capability"][cap_name] = {
                    "wins": stats["wins"],
                    "ties": stats["ties"],
                    "losses": stats["losses"],
                    "avg_score": round(cap_avg, 2),
                }
            
            model_summary["avg_score"] = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0
            summary["models"][model_name] = model_summary
        
        # Determine overall winner
        model_names = list(summary["models"].keys())
        if len(model_names) == 2:
            m1, m2 = model_names
            if summary["models"][m1]["total_wins"] > summary["models"][m2]["total_wins"]:
                summary["overall_winner"] = m1
            elif summary["models"][m2]["total_wins"] > summary["models"][m1]["total_wins"]:
                summary["overall_winner"] = m2
            else:
                summary["overall_winner"] = "tie"
        
        return summary
    
    def _print_summary(self, console) -> None:
        """Print summary to console"""
        from rich.table import Table
        
        console.print("\n[bold green]═══ Benchmark Complete ═══[/bold green]\n")
        
        # Overall stats table
        table = Table(title="Results Summary")
        table.add_column("Model", style="cyan")
        table.add_column("Wins", justify="right")
        table.add_column("Ties", justify="right")
        table.add_column("Losses", justify="right")
        table.add_column("Avg Score", justify="right")
        
        for model_name, stats in self.summary["models"].items():
            table.add_row(
                model_name,
                str(stats["total_wins"]),
                str(stats["total_ties"]),
                str(stats["total_losses"]),
                f"{stats['avg_score']:.2f}",
            )
        
        console.print(table)
        
        # Winner announcement
        winner = self.summary.get("overall_winner", "tie")
        if winner != "tie":
            console.print(f"\n[bold yellow]🏆 Winner: {winner}[/bold yellow]")
        else:
            console.print(f"\n[bold yellow]🤝 Result: Tie[/bold yellow]")
    
    def get_full_results(self) -> Dict[str, Any]:
        """Get complete benchmark results"""
        return {
            "benchmark_id": self.benchmark_id,
            "benchmark_name": self.config.name,
            "timestamp": self.start_time.isoformat() if self.start_time else None,
            "duration_seconds": (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else None,
            "config": {
                "domain": self.config.domain,
                "capabilities": self.config.capabilities,
                "models": [m.model_dump() for m in self.config.models],
                "judge": self.config.judge.model_dump(),
            },
            "summary": self.summary,
            "detailed_results": self.results if self.config.output.include_raw_responses else [],
        }
    
    def save_results(self, path: Optional[str] = None) -> str:
        """Save results to file"""
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"{self.config.output.directory}/results_{timestamp}.json"
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.get_full_results(), f, indent=2, ensure_ascii=False)
        
        return path
