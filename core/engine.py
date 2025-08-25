# core/engine.py
def simulate_total_balance_series(debts, plan):
    """
    Compute total remaining balance over time given a repayment plan.
    
    debts: list of dicts with keys "name" and "balance"
    plan: repayment plan (list of months with allocations)
    """
    balances = {d["name"]: d["balance"] for d in debts}
    series = [sum(balances.values())]

    for month in plan:
        for debt_name, alloc in month.items():
            # Subtract principal from debt balance
            balances[debt_name] = max(0, balances[debt_name] - alloc.get("principal", 0.0))
        series.append(sum(balances.values()))
    
    return series
