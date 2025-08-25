# core/scenarios.py
from typing import List, Dict, Any
from .schemas import Debt
from .optimization import compute_avalanche_plan, compute_snowball_plan, one_step_optimal_allocation

def simulate_payoff(debts: List[Debt], base_budget: float, extra_payment: float=0.0,
                    consolidation_apr: float=0.0, months: int=36) -> Dict[str, Any]:
    ds = [Debt(**d.model_dump()) for d in debts]
    if consolidation_apr and consolidation_apr > 0:
        # consolidation_apr may be percent (e.g., 12) or decimal (0.12). normalize:
        apr = consolidation_apr
        if apr > 1.0:
            apr = apr / 100.0
        for d in ds:
            d.apr = apr

    budget = base_budget + extra_payment
    aval = compute_avalanche_plan(ds, budget, months)
    snow = compute_snowball_plan(ds, budget, months)
    one = one_step_optimal_allocation(ds, budget)
    # pick best by total interest then months to debt free
    candidates = [aval, snow, one]
    best = min(candidates, key=lambda p: (p.total_interest_paid if p.total_interest_paid is not None else 1e12, p.months_to_debt_free if p.months_to_debt_free>0 else 1e9))
    return {
        "budget_used": budget,
        "avalanche": aval.model_dump(),
        "snowball": snow.model_dump(),
        "one_step_optimal": one.model_dump(),
        "best_plan": best.strategy
    }
