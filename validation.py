"""
Validation module
Reconciles extracted amounts with bill total
"""

import re
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class AmountValidator:
    """
    Validates and reconciles extracted amounts
    """
    
    def __init__(self):
        logger.info("Initialized AmountValidator")
    
    def validate(self, line_items: List[Dict], full_text: str) -> Dict:
        """
        Validate extracted amounts against bill total
        
        Args:
            line_items: Extracted line items
            full_text: Complete OCR text
            
        Returns:
            Validation result with reconciliation status
        """
        logger.info("Validating amounts")
        
        # Calculate extracted total
        extracted_total = sum(item['item_amount'] for item in line_items)
        
        # Find bill total
        bill_total = self._find_bill_total(full_text)
        
        if bill_total is None:
            logger.warning("Could not find bill total in text")
            return {
                'is_reconciled': False,
                'extracted_total': extracted_total,
                'bill_total': None,
                'line_items': line_items,
                'error': 'Bill total not found'
            }
        
        # Calculate discrepancy
        discrepancy = abs(extracted_total - bill_total)
        tolerance = 0.01  # 1 paisa tolerance
        
        is_reconciled = discrepancy <= tolerance
        
        logger.info(f"Extracted: {extracted_total}, Bill total: {bill_total}, Reconciled: {is_reconciled}")
        
        return {
            'is_reconciled': is_reconciled,
            'extracted_total': extracted_total,
            'bill_total': bill_total,
            'discrepancy': discrepancy,
            'line_items': line_items
        }
    
    def _find_bill_total(self, text: str) -> float:
        """
        Find grand total in bill text
        
        Args:
            text: Full OCR text
            
        Returns:
            Total amount or None
        """
        patterns = [
            r'Grand\s*Total\s*[:=]?\s*₹?\s*(\d+[,\d]*\.?\d*)',
            r'Total\s*Amount\s*[:=]?\s*₹?\s*(\d+[,\d]*\.?\d*)',
            r'Net\s*Amount\s*[:=]?\s*₹?\s*(\d+[,\d]*\.?\d*)',
            r'Amount\s*Due\s*[:=]?\s*₹?\s*(\d+[,\d]*\.?\d*)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Take last match
                amount_str = matches[-1].replace(',', '')
                try:
                    return float(amount_str)
                except:
                    pass
        
        return None
