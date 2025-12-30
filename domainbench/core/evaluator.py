"""
Evaluator - LLM-as-Judge evaluation with swap mitigation
Based on the judge logic from waiterbench.py
"""

import json
from typing import List, Dict, Any, Optional

from domainbench.providers.base import BaseProvider


# Judge prompt returns STRICT JSON for easy parsing
JUDGE_PROMPT_TEMPLATE = """You are evaluating two assistant responses in the role of: {role}

Pick the better response based on:
1) Task accuracy & completeness
2) Safe handling of constraints and edge cases
3) Appropriate clarifying questions (ask when needed; avoid excessive questions)
4) Natural, helpful tone (polite, concise, professional)
5) Memory across turns (no contradictions; respects earlier constraints)
6) Actionability (clear next steps; offers alternatives when needed)

Return STRICT JSON (no markdown), schema exactly:
{{"winner":"A"|"B"|"tie","score_A":0-10,"score_B":0-10,"reasons":[string,...]}}

Conversation (multi-turn user messages):
{conversation}

Response A:
{response_a}

Response B:
{response_b}
"""


def safe_json_loads(s: str) -> Optional[dict]:
    """Best-effort strict JSON parsing; returns None if not parseable."""
    # Try to extract JSON from potential markdown code blocks
    if "```json" in s:
        start = s.find("```json") + 7
        end = s.find("```", start)
        if end > start:
            s = s[start:end].strip()
    elif "```" in s:
        start = s.find("```") + 3
        end = s.find("```", start)
        if end > start:
            s = s[start:end].strip()
    
    try:
        return json.loads(s)
    except Exception:
        return None


def normalize_judge_result(obj: dict) -> dict:
    """Coerce judge JSON into expected schema; fallback to tie on invalid."""
    if not isinstance(obj, dict):
        return {"winner": "tie", "score_A": 0, "score_B": 0, "reasons": ["Invalid judge output (not a dict)."]}

    winner = obj.get("winner", "tie")
    if winner not in ("A", "B", "tie"):
        winner = "tie"

    def clamp_score(x):
        try:
            x = float(x)
        except Exception:
            x = 0.0
        if x < 0:
            x = 0.0
        if x > 10:
            x = 10.0
        return x

    score_a = clamp_score(obj.get("score_A", 0))
    score_b = clamp_score(obj.get("score_B", 0))
    reasons = obj.get("reasons", [])
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    reasons = [str(r) for r in reasons][:8]  # cap verbosity

    return {"winner": winner, "score_A": score_a, "score_B": score_b, "reasons": reasons}


def swap_mitigated_winner(j_ab: dict, j_ba: dict) -> str:
    """
    Apply swap-order mitigation to determine final winner.
    
    j_ab: judge(A vs B)
    j_ba: judge(B vs A)
    
    Map j_ba back to original labels:
      - if j_ba says A, it means original B
      - if j_ba says B, it means original A
    If inconsistent => tie
    """
    w1 = j_ab["winner"]
    w2 = j_ba["winner"]
    
    if w2 == "A":
        w2_mapped = "B"
    elif w2 == "B":
        w2_mapped = "A"
    else:
        w2_mapped = "tie"
    
    return w1 if w1 == w2_mapped else "tie"


class Evaluator:
    """Base evaluator interface"""
    
    def evaluate(self, response: str, expected: Any) -> Dict[str, Any]:
        """Evaluate a single response"""
        raise NotImplementedError


class JudgeEvaluator(Evaluator):
    """
    LLM-as-Judge evaluator with swap-order mitigation.
    
    Uses another LLM to compare two responses and determine which is better.
    Runs comparison twice with swapped order to mitigate position bias.
    """
    
    def __init__(self, provider: BaseProvider, model: str, max_retries: int = 2):
        self.provider = provider
        self.model = model
        self.max_retries = max_retries
    
    def evaluate_pair(
        self,
        conversation: List[str],
        response_a: str,
        response_b: str,
        system_prompt: str = "",
        role: str = "a helpful assistant",
    ) -> Dict[str, Any]:
        """
        Evaluate two responses with swap-order mitigation.
        
        Returns:
            Dict with winner ("A", "B", or "tie"), scores, and reasons
        """
        # Format conversation
        conv_text = "\n".join([f"USER[{i+1}]: {t}" for i, t in enumerate(conversation)])
        
        # First comparison: A vs B
        j_ab = self._judge_once(conv_text, response_a, response_b, role)
        
        # Second comparison: B vs A (swapped)
        j_ba = self._judge_once(conv_text, response_b, response_a, role)
        
        # Apply swap mitigation
        final_winner = swap_mitigated_winner(j_ab, j_ba)
        
        # Average scores (accounting for swap)
        avg_score_a = (j_ab["score_A"] + j_ba["score_B"]) / 2
        avg_score_b = (j_ab["score_B"] + j_ba["score_A"]) / 2
        
        # Combine reasons
        all_reasons = j_ab["reasons"] + j_ba["reasons"]
        unique_reasons = list(dict.fromkeys(all_reasons))[:6]  # Dedupe and limit
        
        return {
            "winner": final_winner,
            "score_A": round(avg_score_a, 1),
            "score_B": round(avg_score_b, 1),
            "reasons": unique_reasons,
            "raw_ab": j_ab,
            "raw_ba": j_ba,
        }
    
    def _judge_once(
        self,
        conversation: str,
        response_a: str,
        response_b: str,
        role: str,
    ) -> dict:
        """Run a single judge comparison"""
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            role=role,
            conversation=conversation,
            response_a=response_a,
            response_b=response_b,
        )
        
        messages = [{"role": "user", "content": prompt}]
        last_text = ""
        
        for attempt in range(self.max_retries + 1):
            response = self.provider.chat_completion(
                model=self.model,
                messages=messages,
                temperature=0.0,
            )
            text = response.get("content", "")
            last_text = text
            
            obj = safe_json_loads(text)
            if obj is not None:
                return normalize_judge_result(obj)
            
            # Nudge to output strict JSON only
            messages.append({"role": "assistant", "content": text})
            messages.append({
                "role": "user", 
                "content": "Your previous output was not valid JSON. Output ONLY strict JSON per the schema."
            })
        
        return {
            "winner": "tie",
            "score_A": 0,
            "score_B": 0,
            "reasons": [f"Judge output not parseable as JSON. Last: {last_text[:200]}"]
        }


class RuleBasedEvaluator(Evaluator):
    """
    Simple rule-based evaluator for structured outputs.
    Useful for function calling, JSON schema validation, etc.
    """
    
    def __init__(self, rules: List[Dict[str, Any]] = None):
        self.rules = rules or []
    
    def evaluate(self, response: str, expected: Any) -> Dict[str, Any]:
        """Evaluate response against expected value using rules"""
        score = 0.0
        reasons = []
        
        # Exact match check
        if expected and response.strip() == str(expected).strip():
            score = 1.0
            reasons.append("Exact match")
        
        # Keyword presence check
        if isinstance(expected, dict) and "keywords" in expected:
            keywords = expected["keywords"]
            found = sum(1 for kw in keywords if kw.lower() in response.lower())
            keyword_score = found / len(keywords) if keywords else 0
            score = max(score, keyword_score)
            reasons.append(f"Keywords: {found}/{len(keywords)}")
        
        return {
            "score": score,
            "passed": score >= 0.5,
            "reasons": reasons,
        }
