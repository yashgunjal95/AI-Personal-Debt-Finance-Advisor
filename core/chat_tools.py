# core/chat_tools.py
import json
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
from .schemas import Debt, UserProfile
from .optimization import (
    compute_avalanche_plan,
    compute_snowball_plan,
    one_step_optimal_allocation,
)
from .plan_utils import plan_to_dataframe, simulate_total_balance_series
from .education import rag_answer

# ---------- Parsing & Presentation ----------

def parse_debts_json(text: str) -> Tuple[List[Debt], Optional[str]]:
    try:
        data = json.loads(text)
        if not isinstance(data, list):
            return [], "Debts JSON must be a list of objects."
        debts = [Debt(**d) for d in data]
        return debts, None
    except Exception as e:
        return [], f"Invalid debts JSON: {e}"

def pretty_debts_table(debts: List[Debt]) -> pd.DataFrame:
    rows = []
    for d in debts:
        rows.append({
            "Debt": d.name,
            "Balance (₹)": float(d.balance),
            "APR (%)": float(d.apr) * 100.0,
            "Min Payment (₹)": float(d.min_payment),
            "Est. Monthly Interest (₹)": float(d.balance) * float(d.apr) / 12.0,
            "Limit (₹)": float(getattr(d, "limit", 0.0)) if getattr(d, "limit", None) else None
        })
    return pd.DataFrame(rows)

def summarize_debts(debts: List[Debt], profile: Optional[UserProfile] = None) -> str:
    if not debts:
        return "No debts provided."
    total_bal = sum(d.balance for d in debts)
    total_min = sum(d.min_payment for d in debts if d.balance > 0)
    w_apr = 0.0
    if total_bal > 0:
        w_apr = sum(d.apr * d.balance for d in debts) / total_bal
    lines = [
        f"Total debts: ₹{total_bal:,.0f}",
        f"Weighted APR: {w_apr*100:.2f}%",
        f"Total minimums: ₹{total_min:,.0f}/month",
    ]
    if profile:
        leftover = profile.monthly_income - profile.monthly_expenses - total_min
        lines.append(f"Estimated leftover after mins: ₹{leftover:,.0f}/month")
    return "\n".join(lines)

# ---------- Strategy Runner ----------

def run_plan_tool(
    debts: List[Debt],
    budget: float,
    months: int,
    strategy: str = "avalanche"
) -> Dict[str, Any]:
    strat = strategy.strip().lower()
    if strat.startswith("snow"):
        plan = compute_snowball_plan(debts, budget, months)
        name = "Debt Snowball"
    elif strat.startswith("one") or strat.startswith("opt") or strat == "lp":
        plan = one_step_optimal_allocation(debts, budget)
        name = "One-Step Optimal (LP)"
    else:
        plan = compute_avalanche_plan(debts, budget, months)
        name = "Debt Avalanche"

    df = plan_to_dataframe(plan)
    series = simulate_total_balance_series(debts, plan)
    summary = {
        "strategy_name": name,
        "months": len(plan.months),
        "total_interest": sum(m.total_interest for m in plan.months),
        "plan": plan,
        "schedule_df": df,
        "balance_series": series,
    }
    return summary

def compare_baseline_vs_extra(
    debts: List[Debt],
    budget: float,
    months: int,
    extra: float = 0.0,
    strategy: str = "avalanche"
) -> Dict[str, Any]:
    base = run_plan_tool(debts, budget, months, strategy=strategy)
    scenario = run_plan_tool(debts, budget + max(0.0, extra), months, strategy=strategy)
    base_int = base["total_interest"]
    scen_int = scenario["total_interest"]
    base_m = base["months"]
    scen_m = scenario["months"]
    return {
        "baseline": base,
        "scenario": scenario,
        "interest_savings": max(0.0, base_int - scen_int),
        "months_saved": max(0, base_m - scen_m),
    }

# ---------- Slash commands ----------

def run_rag_tool(query: str, vs) -> Tuple[str, List[str]]:
    # non-streaming: used inside Chat tab when user types /rag ...
    answer, sources = rag_answer(query, vs)
    return answer, sources

def parse_slash_command(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Very small parser for messages like:
    /summary
    /plan strategy=avalanche budget=15000 months=60
    /whatif extra=3000 budget=18000 months=48 strategy=snowball
    /rag question="what is snowball vs avalanche"
    """
    if not text.startswith("/"):
        return "", {}
    parts = text.strip().split()
    cmd = parts[0][1:].lower()  # remove leading /
    kv: Dict[str, str] = {}
    # parse key=value (with quotes support)
    rest = text[len(parts[0]):].strip()
    # cheap parse: split by spaces but keep quoted segments
    buf = ""
    in_quotes = False
    for ch in rest:
        if ch == '"':
            in_quotes = not in_quotes
            buf += ch
        elif ch == " " and not in_quotes:
            buf += "\n"
        else:
            buf += ch
    for token in [t for t in buf.split("\n") if t.strip()]:
        if "=" in token:
            k, v = token.split("=", 1)
            v = v.strip().strip('"')
            kv[k.strip().lower()] = v
    return cmd, kv
