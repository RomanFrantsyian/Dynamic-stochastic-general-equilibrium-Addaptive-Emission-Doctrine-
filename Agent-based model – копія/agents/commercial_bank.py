"""
Commercial Bank agent - loan processing, NPL verification.
"""
from typing import List, Dict, TYPE_CHECKING
from dataclasses import dataclass
from agents.base import AEDAgent

if TYPE_CHECKING:
    from model import AEDModel
from data_structures.loan import LoanRecord


class CommercialBank(AEDAgent):
    """
    Commercial bank managing loans and deposits.

    Responsibilities:
    - Loan origination and management
    - NPL identification and verification
    - Deflation verification
    - Interest collection
    - Central bank reporting
    """

    def __init__(self, model: 'AEDModel', initial_reserves: float = 10_000_000):
        super().__init__(model)

        # Capital and reserves
        self.reserves = initial_reserves
        self.create('money', self.reserves)

        # Loan portfolio
        self.loans: List[LoanRecord] = []
        self.total_loans = 0.0
        self.npl_ratio = 0.0

        # NPL tracking
        self.restructuring_count = 0
        self.debt_annihilated_this_period = 0.0
        self.eligible_for_compensation = False

        # Parameters
        self.base_interest_rate = model.config.get('bank_base_interest_rate', 0.07)
        self.min_capital_ratio = model.config.get('bank_min_capital_ratio', 0.08)

    def process_loan_applications(self):
        """
        Process loan applications from firms.

        Called in sub-round 8 (Investment).
        """
        messages = self.model.message_queue.get_messages(
            recipient_type='commercialbank',
            recipient_id=self.unique_id,
            topic='loan_application'
        )

        for msg in messages:
            borrower_id = msg['sender_id']
            borrower_type = msg['sender_type']
            amount = msg['content']['amount']

            if self._can_lend(amount):
                loan = LoanRecord(
                    loan_id=len(self.loans),
                    borrower_id=borrower_id,
                    borrower_type=borrower_type,
                    principal=amount,
                    interest_rate=self.base_interest_rate,
                    is_performing=True
                )
                self.loans.append(loan)
                self.total_loans += amount

                borrower = self.model.agent_registry.get_agent(borrower_type, borrower_id)
                if borrower:
                    self.give(borrower, 'money', amount)

                    self.model.message_queue.send(
                        sender_type='commercialbank',
                        sender_id=self.unique_id,
                        recipient_type=borrower_type,
                        recipient_id=borrower_id,
                        topic='loan_approved',
                        content={'loan_id': loan.loan_id, 'amount': amount}
                    )

    def _can_lend(self, amount: float) -> bool:
        """Check if bank has capacity to lend."""
        capital_ratio = self.reserves / (self.total_loans + amount) if (self.total_loans + amount) > 0 else 1.0
        return capital_ratio >= self.min_capital_ratio and self['money'] >= amount

    def verify_deflation_claims(self):
        """
        Verify deflation declarations from borrowers.
        Passes patent info to central bank for 70/20/10 distribution.

        Called in sub-round 6 (Deflation Verification).
        """
        messages = self.model.message_queue.get_messages(
            recipient_type='commercialbank',
            recipient_id=self.unique_id,
            topic='deflation_declaration'
        )

        self.debt_annihilated_this_period = 0.0
        npls_verified = 0
        patent_vacuums = []  # list of {implementer_id, innovator_id, patent_id, vacuum}

        for msg in messages:
            borrower_id = msg['sender_id']
            borrower_type = msg['sender_type']
            content = msg['content']
            declared_deflation = content.get('production_cost_reduction', 0.0)
            price_reduction = content.get('price_reduction', 0.0)
            patent_id = content.get('patent_id')
            innovator_id = content.get('innovator_id')

            for loan in self.loans:
                if loan.borrower_id == borrower_id and loan.borrower_type == borrower_type:
                    if self._verify_deflation(loan, declared_deflation):
                        loan.is_performing = False
                        loan.deflation_declared = True
                        self.debt_annihilated_this_period += loan.principal
                        self.total_loans -= loan.principal
                        npls_verified += 1

                        # Reduce firm's debt
                        borrower = self.model.agent_registry.get_agent(borrower_type, borrower_id)
                        if borrower:
                            borrower.debt = max(0.0, borrower.debt - loan.principal)

                        # Collect patent vacuum info for 70/20/10 distribution
                        if price_reduction > 0 and patent_id is not None and innovator_id is not None:
                            patent_vacuums.append({
                                'implementer_id': borrower_id,
                                'innovator_id': innovator_id,
                                'patent_id': patent_id,
                                'price_reduction': price_reduction
                            })

        # Report to central bank
        if npls_verified > 0:
            self.model.message_queue.send(
                sender_type='commercialbank',
                sender_id=self.unique_id,
                recipient_type='centralbank',
                recipient_id=self.model.central_bank.unique_id,
                topic='debt_restructuring',
                content={
                    'debt_annihilated': self.debt_annihilated_this_period,
                    'npl_count': npls_verified,
                    'deflation_verified': True,
                    'patent_vacuums': patent_vacuums  # new: patent-level vacuum data
                }
            )

        # Remove annihilated loans from portfolio
        self.loans = [l for l in self.loans if l.is_performing]

        self.npl_ratio = sum(1 for l in self.loans if not l.is_performing) / len(self.loans) if self.loans else 0.0

    def _verify_deflation(self, loan: LoanRecord, declared_deflation: float) -> bool:
        """
        Verify deflation claim using fraud detection heuristics.

        Returns True if deflation is legitimate.
        """
        from mechanisms.verification import verify_deflation_claim

        borrower = self.model.agent_registry.get_agent(loan.borrower_type, loan.borrower_id)
        if not borrower:
            return False

        return verify_deflation_claim(
            borrower=borrower,
            declared_deflation=declared_deflation,
            fraud_threshold=self.model.central_bank.fraud_detection_threshold
        )

    def collect_interest(self):
        """
        Collect interest payments from performing loans.

        Called in sub-round 5 (Wage Payments & Taxes).
        """
        for loan in self.loans:
            if loan.is_performing:
                borrower = self.model.agent_registry.get_agent(loan.borrower_type, loan.borrower_id)
                if borrower:
                    interest = loan.principal * loan.interest_rate
                    if borrower.has_goods('money', interest):
                        borrower.give(self, 'money', interest)

    def receive_emission_compensation(self):
        """
        Receive compensation from central bank for verified NPLs.

        Called in sub-round 7 (Debt Restructuring & Emission).
        """
        if not self.eligible_for_compensation:
            return

        compensation_amount = self.debt_annihilated_this_period * self.model.central_bank.emission_coverage_ratio

        if compensation_amount > 0:
            self.create('money', compensation_amount)
            self.reserves += compensation_amount
            self.restructuring_count += 1

        self.eligible_for_compensation = False