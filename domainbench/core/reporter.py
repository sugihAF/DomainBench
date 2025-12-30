"""
Reporter - Generate benchmark reports in various formats
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from domainbench.core.config import OutputConfig


class Reporter:
    """
    Generate benchmark reports in various formats.
    
    Supports:
    - JSON: Machine-readable full results
    - Markdown: GitHub-friendly summary
    - HTML: Visual dashboard (future)
    """
    
    def __init__(self, config: OutputConfig):
        self.config = config
        self.output_dir = Path(config.directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save(
        self,
        results: Dict[str, Any],
        formats: Optional[List[str]] = None,
        base_name: Optional[str] = None,
    ) -> List[str]:
        """
        Save results in specified formats.
        
        Args:
            results: Full benchmark results dictionary
            formats: List of formats to save (defaults to config)
            base_name: Base filename (without extension)
            
        Returns:
            List of saved file paths
        """
        formats = formats or self.config.formats
        if base_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"results_{timestamp}"
        
        saved_paths = []
        
        for fmt in formats:
            if fmt == "json":
                path = self._save_json(results, base_name)
            elif fmt == "markdown" or fmt == "md":
                path = self._save_markdown(results, base_name)
            elif fmt == "jsonl":
                path = self._save_jsonl(results, base_name)
            else:
                continue
            
            saved_paths.append(path)
        
        return saved_paths
    
    def _save_json(self, results: Dict[str, Any], base_name: str) -> str:
        """Save full results as JSON"""
        path = self.output_dir / f"{base_name}.json"
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        return str(path)
    
    def _save_jsonl(self, results: Dict[str, Any], base_name: str) -> str:
        """Save detailed results as JSONL (one result per line)"""
        path = self.output_dir / f"{base_name}.jsonl"
        
        with open(path, 'w', encoding='utf-8') as f:
            # Write summary as first line
            f.write(json.dumps({
                "type": "summary",
                "benchmark_id": results.get("benchmark_id"),
                "benchmark_name": results.get("benchmark_name"),
                "timestamp": results.get("timestamp"),
                "summary": results.get("summary"),
            }, ensure_ascii=False, default=str) + '\n')
            
            # Write each detailed result
            for result in results.get("detailed_results", []):
                f.write(json.dumps({
                    "type": "result",
                    **result,
                }, ensure_ascii=False, default=str) + '\n')
        
        return str(path)
    
    def _save_markdown(self, results: Dict[str, Any], base_name: str) -> str:
        """Generate markdown report"""
        path = self.output_dir / f"{base_name}.md"
        
        summary = results.get("summary", {})
        models = summary.get("models", {})
        config = results.get("config", {})
        
        md_lines = [
            f"# Benchmark Report: {results.get('benchmark_name', 'Unknown')}",
            "",
            f"**Date:** {results.get('timestamp', 'Unknown')}",
            f"**Benchmark ID:** `{results.get('benchmark_id', 'Unknown')}`",
            f"**Duration:** {results.get('duration_seconds', 0):.1f} seconds",
            "",
            "## Configuration",
            "",
            f"- **Domain:** {config.get('domain', 'Unknown')}",
            f"- **Capabilities:** {', '.join(config.get('capabilities', []))}",
            f"- **Judge Model:** {config.get('judge', {}).get('model', 'Unknown')}",
            "",
            "### Models",
            "",
        ]
        
        for model in config.get("models", []):
            md_lines.append(f"- **{model.get('alias', model.get('model'))}**: {model.get('provider')}/{model.get('model')}")
        
        md_lines.extend([
            "",
            "## Results Summary",
            "",
            "| Model | Wins | Ties | Losses | Avg Score |",
            "|-------|------|------|--------|-----------|",
        ])
        
        for model_name, stats in models.items():
            md_lines.append(
                f"| {model_name} | {stats.get('total_wins', 0)} | "
                f"{stats.get('total_ties', 0)} | {stats.get('total_losses', 0)} | "
                f"{stats.get('avg_score', 0):.2f} |"
            )
        
        winner = summary.get("overall_winner", "tie")
        md_lines.extend([
            "",
            f"**Overall Winner:** {'🏆 ' + winner if winner != 'tie' else '🤝 Tie'}",
            "",
        ])
        
        # Category breakdown
        by_category = summary.get("by_category", {})
        if by_category:
            md_lines.extend([
                "## Results by Category",
                "",
                "| Category | " + " | ".join(models.keys()) + " |",
                "|----------|" + "|".join(["---"] * len(models)) + "|",
            ])
            
            for cat, cat_stats in by_category.items():
                row = f"| {cat} |"
                for model_name in models.keys():
                    m_stats = cat_stats.get(model_name, {})
                    wins = m_stats.get("wins", 0)
                    ties = m_stats.get("ties", 0)
                    row += f" W:{wins} T:{ties} |"
                md_lines.append(row)
            
            md_lines.append("")
        
        md_lines.extend([
            "---",
            f"*Generated by DomainBench*",
        ])
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_lines))
        
        return str(path)
    
    @staticmethod
    def generate_summary_table(results: Dict[str, Any]) -> str:
        """Generate a simple ASCII table summary"""
        summary = results.get("summary", {})
        models = summary.get("models", {})
        
        lines = [
            "=" * 60,
            f"  Benchmark: {results.get('benchmark_name', 'Unknown')}",
            "=" * 60,
            "",
            f"{'Model':<30} {'Wins':>8} {'Ties':>8} {'Score':>8}",
            "-" * 60,
        ]
        
        for model_name, stats in models.items():
            lines.append(
                f"{model_name:<30} {stats.get('total_wins', 0):>8} "
                f"{stats.get('total_ties', 0):>8} {stats.get('avg_score', 0):>8.2f}"
            )
        
        lines.extend([
            "-" * 60,
            f"Winner: {summary.get('overall_winner', 'tie')}",
            "=" * 60,
        ])
        
        return '\n'.join(lines)
