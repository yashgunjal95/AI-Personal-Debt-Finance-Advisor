#tests/test_optimization.py
from core.schemas import Debt
from core.optimization import compute_avalanche_plan, compute_snowball_plan

def sample_debts():
    return [
        Debt(name="High APR Card", balance=120000, apr=0.36, min_payment=3000),
        Debt(name="Low APR Loan", balance=240000, apr=0.12, min_payment=2500),
        Debt(name="Medium APR Loan", balance=80000, apr=0.18, min_payment=2000),
    ]

def test_avalanche_interest_not_worse_than_snowball():
    debts = sample_debts()
    budget = 20000
    months = 36
    aval = compute_avalanche_plan(debts, budget, months)
    snow = compute_snowball_plan(debts, budget, months)
    assert aval.total_interest_paid <= snow.total_interest_paid + 1e-6

def test_budget_guardrail():
    debts = sample_debts()
    # budget below minimum payments (3000+2500+2000=7500)
    aval = compute_avalanche_plan(debts, 1000, 12)
    assert aval.months_to_debt_free == 0 and len(aval.months) == 0
