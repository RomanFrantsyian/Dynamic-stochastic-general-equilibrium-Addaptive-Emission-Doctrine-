"""
Loan data structure.
"""
from dataclasses import dataclass


@dataclass
class LoanRecord:
    """Loan record structure."""
    loan_id: int
    borrower_id: int
    borrower_type: str
    principal: float
    interest_rate: float
    is_performing: bool
    deflation_declared: bool = False
