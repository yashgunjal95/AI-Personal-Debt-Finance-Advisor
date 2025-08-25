# core/utils.py
def money(x: float) -> str:
    try:
        return f"₹{x:,.0f}"
    except Exception:
        return f"₹{x}"

def month_year_iter(start_month=1, start_year=2025, months=12):
    m, y = start_month, start_year
    for _ in range(months):
        yield m, y
        m += 1
        if m > 12:
            m = 1
            y += 1
