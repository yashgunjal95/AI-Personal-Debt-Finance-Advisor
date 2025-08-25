# core/plan_utils.py
from typing import List
import pandas as pd
from .schemas import Debt, RepaymentPlan
from .optimization import _monthly_rate

def plan_to_dataframe(plan: RepaymentPlan) -> pd.DataFrame:
    rows = []
    for m in plan.months:
        for a in m.allocations:
            rows.append({
                "month": m.month_index,
                "debt": a.name,
                "payment": a.payment,
                "interest": a.interest_accrued,
                "principal": a.principal_reduction,
                "total_paid_month": m.total_paid,
                "month_interest_total": m.total_interest,
            })
    if not rows:
        return pd.DataFrame(columns=["month","debt","payment","interest","principal","total_paid_month","month_interest_total"])
    return pd.DataFrame(rows)

def simulate_total_balance_series(initial_debts: List[Debt], plan: RepaymentPlan) -> List[float]:
    ds = [Debt(**d.model_dump()) for d in initial_debts]
    totals = []
    for m in plan.months:
        name_to = {d.name: d for d in ds}
        for a in m.allocations:
            d = name_to.get(a.name)
            if not d or d.balance <= 0:
                continue
            r = _monthly_rate(d.apr)
            interest = d.balance * r
            due = d.balance + interest
            pay = min(a.payment, due)
            d.balance = max(0.0, due - pay)
        totals.append(sum(d.balance for d in ds))
    return totals
