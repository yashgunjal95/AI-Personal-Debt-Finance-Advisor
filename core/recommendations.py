# core/recommendations.py
from typing import List, Dict, Any, Optional, Tuple
from .schemas import Debt, UserProfile
from .utils import money
import math

HIGH_APR_THRESHOLD = 0.15
MEDIUM_APR_THRESHOLD = 0.08
UTILIZATION_TARGET = 0.30
EMERGENCY_FUND_MONTHS = 3

def _monthly_rate(apr: float) -> float:
    return max(0.0, apr) / 12.0

def _simulate_interest_for_debt(debt: Debt, monthly_payment: float, months: int) -> Tuple[float,int]:
    bal = float(debt.balance)
    r = _monthly_rate(float(debt.apr))
    total_interest = 0.0
    m = 0
    max_iter = months
    while m < max_iter and bal > 0.01:
        interest = bal * r
        pay = min(monthly_payment, bal + interest)
        principal = max(0.0, pay - interest)
        bal = max(0.0, bal + interest - pay)
        total_interest += interest
        m += 1
        if monthly_payment <= interest and m >= 12:
            break
    return total_interest, m

def estimate_savings_by_extra(debt: Debt, extra_monthly: float, horizon_months: int = 12) -> float:
    base_payment = max(debt.min_payment, 0.0)
    base_interest, _ = _simulate_interest_for_debt(debt, base_payment, horizon_months)
    new_payment = base_payment + extra_monthly
    new_interest, _ = _simulate_interest_for_debt(debt, new_payment, horizon_months)
    return max(0.0, base_interest - new_interest)

def _score_debt_for_priority(debt: Debt) -> float:
    apr_factor = float(debt.apr)
    balance_factor = math.log1p(max(0.0, debt.balance))
    minpay_factor = float(debt.min_payment) / max(1.0, debt.balance)
    score = apr_factor * 3.0 + (balance_factor * 0.3) + (minpay_factor * 2.0)
    return score

def generate_recommendations(profile: UserProfile,
                             debts: List[Debt],
                             extra_budget: Optional[float] = None,
                             top_n: int = 6) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    total_min_payments = sum(d.min_payment for d in debts if d.balance > 0)
    total_revolving_balances = sum(d.balance for d in debts)
    income = float(getattr(profile, "monthly_income", getattr(profile, "income", 0.0)))
    expenses = float(getattr(profile, "monthly_expenses", getattr(profile, "expenses", 0.0)))
    savings_buffer = max(0.0, income - expenses - total_min_payments)
    scored = []
    for d in debts:
        if d.balance <= 0.01:
            continue
        sc = _score_debt_for_priority(d)
        scored.append((sc, d))
    scored.sort(reverse=True, key=lambda x: x[0])
    alloc_budget = float(extra_budget) if extra_budget is not None else max(0.0, savings_buffer)
    for idx, (sc, d) in enumerate(scored):
        title = f"Pay extra toward '{d.name}'"
        extra_alloc = 0.0
        if alloc_budget > 0:
            weight = 0.6 if d.apr >= HIGH_APR_THRESHOLD else 0.4
            extra_alloc = min(alloc_budget * weight, max(0.0, d.balance))
        est_saving = estimate_savings_by_extra(d, extra_alloc, horizon_months=12) if extra_alloc > 0 else 0.0
        explanation = (
            f"{'High APR' if d.apr>=HIGH_APR_THRESHOLD else 'Consider extra payments'}: "
            f"APR {d.apr*100:.1f}%, balance {money(d.balance)}. "
            f"Paying an additional {money(extra_alloc)}/month could save ~{money(est_saving)} interest over 12 months (approx)."
        )
        recs.append({
            "id": f"pay_extra::{d.name}",
            "type": "repayment",
            "title": title,
            "score": float(sc + (d.apr * 10)),
            "action": f"Allocate extra ₹{int(extra_alloc):,}/month to {d.name}.",
            "estimated_savings": est_saving,
            "explanation": explanation,
            "debt_name": d.name,
            "suggested_extra_monthly": extra_alloc
        })
    avg_apr = sum(d.apr * d.balance for d in debts if d.balance > 0) / max(1.0, total_revolving_balances) if total_revolving_balances>0 else 0.0
    if avg_apr >= HIGH_APR_THRESHOLD and total_revolving_balances > 0:
        recs.append({
            "id": "consolidation::consider",
            "type": "refinance",
            "title": "Consider debt consolidation / refinancing",
            "score": 8.0 + float(avg_apr*10),
            "action": "Investigate a consolidation loan at APR lower than your weighted average APR.",
            "estimated_savings": 0.0,
            "explanation": (
                f"Weighted average APR across debts is {avg_apr*100:.2f}%. If you can obtain a consolidation loan with a lower APR, "
                "you may reduce total interest and simplify payments. Compare fees and prepayment penalties."
            )
        })
    emergency_target = expenses * EMERGENCY_FUND_MONTHS
    current_surplus = max(0.0, income - expenses - total_min_payments)
    if current_surplus < emergency_target * 0.25:
        recs.append({
            "id": "cash::build_emergency",
            "type": "cashflow",
            "title": "Build an emergency fund",
            "score": 7.0,
            "action": f"Target an emergency fund of {EMERGENCY_FUND_MONTHS} months ({money(emergency_target)}).",
            "estimated_savings": 0.0,
            "explanation": (
                "A 3-month emergency fund prevents new high-interest borrowing if an expense emerges. "
                "If you don't have 3 months of expenses saved, start by allocating a small monthly amount to a liquid savings account."
            )
        })
    revolving_with_limit = [d for d in debts if getattr(d, "limit", None)]
    if revolving_with_limit:
        total_limits = sum(getattr(d, "limit", 0.0) for d in revolving_with_limit)
        util = sum(d.balance for d in revolving_with_limit) / max(1.0, total_limits)
        if util > UTILIZATION_TARGET:
            recs.append({
                "id": "credit::utilization",
                "type": "credit",
                "title": "Reduce credit utilization",
                "score": 6.5,
                "action": f"Reduce revolving balances or request higher limits to target utilization <= {int(UTILIZATION_TARGET*100)}%.",
                "estimated_savings": 0.0,
                "explanation": (
                    f"Your estimated utilization is {util*100:.1f}%. High utilization can lower credit scores; reducing to under 30% helps."
                )
            })
    recs.append({
        "id": "behaviour::autopay",
        "type": "behaviour",
        "title": "Set up autopay for minimums",
        "score": 6.0,
        "action": "Enable autopay for at least the minimum payments to protect on-time history.",
        "estimated_savings": 0.0,
        "explanation": "Consistent on-time payments are the most important factor for credit scores and avoid late fees."
    })
    if income > 0 and (total_min_payments / income) > 0.4:
        recs.append({
            "id": "cash::restructure",
            "type": "cashflow",
            "title": "Debt restructuring / negotiate terms",
            "score": 8.5,
            "action": "Explore refinancing or renegotiating terms with lenders to lower monthly burden.",
            "estimated_savings": 0.0,
            "explanation": (
                f"Your minimum debt obligations ({money(total_min_payments)}) exceed 40% of monthly income ({money(income)}). "
                "Negotiating-term extensions or refinancing can improve monthly cashflow but may increase total interest — model both options."
            )
        })
    recs_sorted = sorted(recs, key=lambda r: r.get("score", 0.0), reverse=True)[:top_n]
    return recs_sorted
