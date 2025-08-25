# core/schemas.py
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Any

class Debt(BaseModel):
    """
    Debt model accepts either:
     - apr (decimal, e.g., 0.36) or
     - interest_rate (percent or decimal; e.g., 36 or 0.36)
    This model normalizes both to:
     - apr (decimal) always available for calculations
     - interest_rate (percent) stored for display convenience
    """
    name: str
    balance: float = Field(ge=0.0)
    apr: Optional[float] = None
    interest_rate: Optional[float] = None  # percent (e.g., 36) or decimal (0.36) - will be normalized
    min_payment: float = Field(ge=0.0)
    limit: Optional[float] = None  # optional credit card limit

    @model_validator(mode="after")
    def normalize_rates(self) -> "Debt":
        # Priority: if apr provided use it. Otherwise if interest_rate present, convert to decimal apr.
        if self.apr is None and self.interest_rate is None:
            # default 0 APR
            self.apr = 0.0
            self.interest_rate = 0.0
            return self

        # If apr given, ensure interest_rate percent is set
        if self.apr is not None:
            # apr is expected decimal like 0.36
            if self.apr > 1.0:
                # guard against user accidentally passing percent in apr field
                self.apr = float(self.apr) / 100.0
            self.interest_rate = float(self.apr) * 100.0
            return self

        # Else interest_rate given: could be 36 (percent) or 0.36 (decimal)
        ir = float(self.interest_rate)
        if ir > 1.0:
            # percent provided
            self.apr = ir / 100.0
            self.interest_rate = ir
        else:
            # decimal provided
            self.apr = ir
            self.interest_rate = ir * 100.0
        return self

    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        # keep pydantic's model_dump but ensure we return normalized fields
        d = super().model_dump(*args, **kwargs)
        # Ensure apr is decimal and interest_rate is percent
        d["apr"] = float(self.apr) if self.apr is not None else 0.0
        d["interest_rate"] = float(self.interest_rate) if self.interest_rate is not None else float(self.apr)*100.0
        return d

class UserProfile(BaseModel):
    # keep names that match your app: monthly_income, monthly_expenses, extra_payment (optional)
    monthly_income: float = 0.0
    monthly_expenses: float = 0.0
    extra_payment: float = 0.0
    # optional debts field for convenience; usually debts are passed separately
    debts: Optional[List[Debt]] = None

    def model_dump(self, *args, **kwargs):
        return super().model_dump(*args, **kwargs)

# Simple types for plan reporting (used by optimization)
class Allocation(BaseModel):
    name: str
    payment: float
    interest_accrued: float
    principal_reduction: float

class RepaymentMonth(BaseModel):
    month_index: int
    allocations: List[Allocation]
    total_interest: float
    total_paid: float

class RepaymentPlan(BaseModel):
    strategy: str
    months: List[RepaymentMonth]
    total_interest_paid: float
    months_to_debt_free: int
