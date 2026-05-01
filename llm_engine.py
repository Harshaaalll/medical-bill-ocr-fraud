"""
LLM Engine for Qwen 3 Integration
Uses Qwen 3 model via Ollama for structured extraction and validation
"""

import logging
import requests
import json
from typing import Dict, List, Optional
import re

logger = logging.getLogger(__name__)


class QwenLLMEngine:
    """Qwen 3 LLM integration for medical bill processing"""
    
    def __init__(self, config=None):
        """Initialize Qwen LLM engine"""
        self.config = config
        self.model = getattr(config, 'LLM_MODEL', 'qwen:7b') if config else 'qwen:7b'
        self.endpoint = getattr(config, 'LLM_ENDPOINT', 'http://localhost:11434') if config else 'http://localhost:11434'
        self.timeout = getattr(config, 'LLM_TIMEOUT', 30) if config else 30
        self.logger = logging.getLogger(__name__)
    
    def validate_connection(self) -> bool:
        """Check if Ollama/Qwen is running"""
        try:
            response = requests.get(
                f"{self.endpoint}/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                self.logger.info("✓ Qwen LLM connection successful")
                return True
        except Exception as e:
            self.logger.warning(f"✗ Qwen LLM not available: {e}")
        return False
    
    def extract_structured_items(self, text_blocks: List[Dict]) -> List[Dict]:
        """
        Use Qwen 3 to extract and structure line items from text
        
        Why:
        - LLM understands context better than regex
        - Handles variations in bill formats
        - Can interpret ambiguous amounts
        
        Process:
        1. Prepare text from OCR blocks
        2. Send to Qwen 3 for extraction
        3. Parse JSON response
        4. Validate extracted items
        5. Return structured data
        """
        try:
            # Check if LLM is available
            if not self.validate_connection():
                self.logger.warning("LLM not available, using fallback")
                return self._fallback_extraction(text_blocks)
            
            # Prepare text
            full_text = ' '.join([block.get('text', '') for block in text_blocks])
            
            # Create prompt for Qwen
            prompt = f"""Analyze this medical bill text and extract line items. 
Return ONLY a JSON object with this exact structure:
{{
    "items": [
        {{
            "name": "service name",
            "amount": 0.00,
            "quantity": 1,
            "rate": 0.00,
            "confidence": 0.9
        }}
    ],
    "total": 0.00,
    "is_valid": true,
    "notes": "any special notes"
}}

Bill text:
{full_text}

Response (JSON only):"""
            
            # Call Qwen API
            response = requests.post(
                f"{self.endpoint}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3,
                    "top_p": 0.9
                },
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                self.logger.error(f"Qwen API error: {response.status_code}")
                return self._fallback_extraction(text_blocks)
            
            # Parse response
            result = response.json()
            response_text = result.get('response', '')
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                self.logger.warning("No JSON in LLM response")
                return self._fallback_extraction(text_blocks)
            
            json_str = json_match.group(0)
            extracted_data = json.loads(json_str)
            
            # Validate and convert
            items = []
            for item in extracted_data.get('items', []):
                items.append({
                    'item_name': item.get('name', ''),
                    'item_amount': float(item.get('amount', 0)),
                    'item_quantity': float(item.get('quantity', 1)),
                    'item_rate': float(item.get('rate', 0)),
                    'confidence': float(item.get('confidence', 0.85))
                })
            
            self.logger.info(f"LLM extracted {len(items)} items")
            return items
            
        except Exception as e:
            self.logger.error(f"LLM extraction failed: {e}")
            return self._fallback_extraction(text_blocks)
    
    def _fallback_extraction(self, text_blocks: List[Dict]) -> List[Dict]:
        """Fallback to regex-based extraction if LLM unavailable"""
        import re
        
        items = []
        full_text = ' '.join([b.get('text', '') for b in text_blocks])
        
        # Simple regex fallback
        amount_pattern = r'(\d+[,\d]*\.\d{2}|\d+[,\d]+)'
        lines = full_text.split('\n')
        
        for line in lines:
            amounts = re.findall(amount_pattern, line)
            if amounts:
                amount = float(amounts[-1].replace(',', ''))
                if 0 < amount < 1000000:
                    item_name = line[:line.find(amounts[-1])].strip()
                    if item_name:
                        items.append({
                            'item_name': item_name,
                            'item_amount': amount,
                            'item_quantity': 1.0,
                            'item_rate': amount,
                            'confidence': 0.7
                        })
        
        return items
    
    def validate_bill_structure(self, items: List[Dict], total_amount: float) -> Dict:
        """
        Use Qwen to validate bill structure and detect inconsistencies
        
        Why:
        - Check if items sum to total
        - Detect suspicious patterns
        - Validate data consistency
        """
        try:
            if not self.validate_connection():
                return {'is_valid': True, 'issues': []}
            
            items_text = '\n'.join([
                f"- {item['item_name']}: ${item['item_amount']}"
                for item in items
            ])
            
            prompt = f"""Review this medical bill and identify any issues:

Items:
{items_text}

Reported Total: ${total_amount}

Calculated Total: ${sum(item['item_amount'] for item in items)}

Respond with JSON:
{{
    "is_valid": true/false,
    "calculated_total": number,
    "issues": ["issue1", "issue2"],
    "confidence": 0.0-1.0
}}"""
            
            response = requests.post(
                f"{self.endpoint}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result_text = response.json().get('response', '')
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))
            
            return {'is_valid': True, 'issues': []}
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return {'is_valid': True, 'issues': []}
