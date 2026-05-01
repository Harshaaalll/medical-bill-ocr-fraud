import re
import logging

logger = logging.getLogger(__name__)

class BillExtractor:
    """Extract line items from bill text"""
    
    @staticmethod
    def extract_items(ocr_text):
        """Extract line items from OCR text"""
        
        if not ocr_text or not isinstance(ocr_text, str):
            logger.warning("Invalid OCR text")
            return []
        
        items = []
        lines = ocr_text.split('\n')
        
        logger.info(f"Processing {len(lines)} lines")
        
        # Keywords to skip (likely totals or headers)
        skip_keywords = [
            'total', 'grand', 'sub-total', 'subtotal', 'net', 
            'due', 'discount', 'tax', 'gst', 'vat', 'cgst', 'sgst',
            'date:', 'bill no:', 'hospital', 'invoice', 'ref no', 
            'page', 'name:', 'address:', 'phone:', 'email:', 'account'
        ]
        
        for line_num, line in enumerate(lines):
            # Skip empty lines
            if not line.strip() or len(line.strip()) < 5:
                continue
            
            # Skip lines with skip keywords
            lower_line = line.lower()
            if any(kw in lower_line for kw in skip_keywords):
                continue
            
            # Try to extract amount from line
            amount = BillExtractor._extract_amount(line)
            
            if amount and 0 < amount < 1000000:  # Reasonable range
                # Extract item name
                item_name = BillExtractor._extract_item_name(line, amount)
                
                if item_name and len(item_name.strip()) > 2:
                    items.append({
                        'item_name': item_name.strip(),
                        'item_amount': round(amount, 2),
                        'item_rate': round(amount, 2),
                        'item_quantity': 1.0,
                        'confidence': 0.85
                    })
        
        logger.info(f"Extracted {len(items)} items")
        return items
    
    @staticmethod
    def _extract_amount(text):
        """Extract rupee amount from text"""
        
        patterns = [
            r'₹\s*(\d+[,\d]*\.?\d*)',
            r'(\d+[,\d]*\.\d{2})\s*(?:$|rs|rupee)',
            r'(\d+[,\d]{1,}\.?\d*)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Get the last match (usually the amount)
                amount_str = matches[-1].replace(',', '')
                try:
                    amount = float(amount_str)
                    if 0 < amount < 1000000:
                        return amount
                except ValueError:
                    pass
        
        return None
    
    @staticmethod
    def _extract_item_name(line, amount):
        """Extract item name from line"""
        
        amount_str = str(int(amount)) if amount == int(amount) else str(amount)
        
        # Find where amount appears
        idx = line.find(amount_str)
        if idx > 0:
            item_name = line[:idx]
        else:
            item_name = line
        
        # Remove special characters
        item_name = re.sub(r'[₹|:\-_]', ' ', item_name)
        item_name = re.sub(r'\s+', ' ', item_name)
        
        return item_name.strip()
    
    @staticmethod
    def find_bill_total(ocr_text):
        """Find the grand total from bill"""
        
        if not ocr_text:
            return None
        
        patterns = [
            r'Grand\s*Total\s*[:=]?\s*₹?\s*(\d+[,\d]*\.?\d*)',
            r'Total\s*Amount\s*[:=]?\s*₹?\s*(\d+[,\d]*\.?\d*)',
            r'Net\s*Amount\s*[:=]?\s*₹?\s*(\d+[,\d]*\.?\d*)',
            r'Final\s*Amount\s*[:=]?\s*₹?\s*(\d+[,\d]*\.?\d*)',
            r'Amount\s*Due\s*[:=]?\s*₹?\s*(\d+[,\d]*\.?\d*)',
            r'Total\s*Due\s*[:=]?\s*₹?\s*(\d+[,\d]*\.?\d*)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, ocr_text, re.IGNORECASE)
            if matches:
                # Get the last (usually most accurate)
                amount_str = matches[-1].replace(',', '')
                try:
                    return float(amount_str)
                except ValueError:
                    pass
        
        return None
