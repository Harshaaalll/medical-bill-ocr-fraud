"""
Fraud Detection Engine
Detects suspicious patterns and anomalies in medical bills
"""

import logging
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)


class FraudDetectionEngine:
    """Detect fraudulent medical bills"""
    
    def __init__(self, config=None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.past_bills = []  # In production, use database
    
    def detect_amount_anomalies(self, items: List[Dict], total: float) -> Dict:
        """
        Detect unusually high amounts
        
        Why:
        - Some charges may be inflated
        - Outliers indicate fraud risk
        
        Method:
        1. Calculate statistics for each item type
        2. Use IQR (Interquartile Range) to find outliers
        3. Flag items outside 95th percentile
        """
        anomalies = []
        
        try:
            amounts = [item['item_amount'] for item in items]
            
            if len(amounts) < 3:
                return {'has_anomalies': False, 'anomalies': []}
            
            # Calculate percentile
            percentile_95 = np.percentile(amounts, 95)
            percentile_5 = np.percentile(amounts, 5)
            iqr = percentile_95 - percentile_5
            
            # Flag outliers
            for item in items:
                amount = item['item_amount']
                
                # Check if unusually high
                if amount > percentile_95 * 1.5:
                    anomalies.append({
                        'item': item['item_name'],
                        'amount': amount,
                        'expected_range': f"${percentile_5:.2f} - ${percentile_95:.2f}",
                        'severity': 'high' if amount > percentile_95 * 2 else 'medium'
                    })
                    self.logger.warning(f"Amount anomaly detected: {item['item_name']} - ${amount}")
            
            return {
                'has_anomalies': len(anomalies) > 0,
                'anomalies': anomalies,
                'statistics': {
                    'min': float(np.min(amounts)),
                    'max': float(np.max(amounts)),
                    'mean': float(np.mean(amounts)),
                    'median': float(np.median(amounts)),
                    'percentile_95': float(percentile_95)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
            return {'has_anomalies': False, 'anomalies': []}
    
    def check_reconciliation(self, items: List[Dict], reported_total: float) -> Dict:
        """
        Check if items sum to reported total
        
        Why:
        - Fraud often involves rounding errors
        - Mathematical inconsistencies indicate issues
        
        Process:
        1. Sum all item amounts
        2. Compare to reported total
        3. Flag if difference > threshold
        """
        try:
            calculated_total = sum(item['item_amount'] for item in items)
            difference = abs(calculated_total - reported_total)
            difference_percent = (difference / reported_total * 100) if reported_total > 0 else 0
            
            is_reconciled = difference_percent < 1.0  # Allow 1% variance
            
            result = {
                'is_reconciled': is_reconciled,
                'calculated_total': round(calculated_total, 2),
                'reported_total': round(reported_total, 2),
                'difference': round(difference, 2),
                'difference_percent': round(difference_percent, 2)
            }
            
            if not is_reconciled:
                self.logger.warning(f"Reconciliation failed: {difference_percent}% difference")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Reconciliation check failed: {e}")
            return {'is_reconciled': False}
    
    def detect_duplicate_bills(self, items: List[Dict], bill_hash: str) -> Dict:
        """
        Detect if this is a duplicate of a previous bill
        
        Why:
        - Fraud rings submit same bill multiple times
        - Duplicates indicate malicious intent
        
        Process:
        1. Calculate bill hash
        2. Check against past bills
        3. Flag if seen recently
        """
        try:
            # In production, query from database with TTL
            recent_bills = [
                b for b in self.past_bills
                if (datetime.now() - b['timestamp']).days <= (self.config.FRAUD_DUPLICATE_DAYS if self.config else 30)
            ]
            
            for past_bill in recent_bills:
                if past_bill['hash'] == bill_hash:
                    days_ago = (datetime.now() - past_bill['timestamp']).days
                    return {
                        'is_duplicate': True,
                        'last_seen': days_ago,
                        'risk': 'critical'
                    }
            
            return {
                'is_duplicate': False,
                'risk': 'low'
            }
            
        except Exception as e:
            self.logger.error(f"Duplicate check failed: {e}")
            return {'is_duplicate': False}
    
    def analyze_item_patterns(self, items: List[Dict]) -> Dict:
        """
        Analyze patterns in items for suspicious behavior
        
        Why:
        - Legitimate bills have patterns
        - Fraudulent bills have unusual combinations
        
        Checks:
        - Too many items (batch fraud)
        - Duplicate item names
        - Unusual quantities
        - Service combinations
        """
        try:
            patterns = {
                'item_count': len(items),
                'unique_items': len(set(item['item_name'] for item in items)),
                'duplicate_items': 0,
                'high_quantity_items': 0,
                'unusual_combinations': 0
            }
            
            # Check for duplicates
            item_names = [item['item_name'] for item in items]
            patterns['duplicate_items'] = len(item_names) - len(set(item_names))
            
            # Check for unusually high quantities
            for item in items:
                if item.get('item_quantity', 1) > 10:
                    patterns['high_quantity_items'] += 1
            
            # Check for unusual combinations
            # Example: Checkup + 50 Lab Tests + 3 Surgeries = unusual
            service_types = self._categorize_services(items)
            if len(service_types) > 5:
                patterns['unusual_combinations'] = 1
            
            # Calculate fraud risk based on patterns
            risk_score = self._calculate_risk_score(patterns)
            
            return {
                'patterns': patterns,
                'risk_score': risk_score,
                'risk_level': 'high' if risk_score > 0.7 else 'medium' if risk_score > 0.4 else 'low'
            }
            
        except Exception as e:
            self.logger.error(f"Pattern analysis failed: {e}")
            return {'patterns': {}, 'risk_score': 0.0}
    
    def _categorize_services(self, items: List[Dict]) -> set:
        """Categorize services into types"""
        service_types = set()
        keywords = {
            'consultation': ['consult', 'visit', 'checkup'],
            'lab': ['lab', 'test', 'blood', 'urine'],
            'imaging': ['xray', 'ct', 'mri', 'ultrasound'],
            'procedure': ['surgery', 'procedure', 'operation'],
            'medication': ['drug', 'medicine', 'tablet']
        }
        
        for item in items:
            name_lower = item['item_name'].lower()
            for category, keywords_list in keywords.items():
                if any(kw in name_lower for kw in keywords_list):
                    service_types.add(category)
        
        return service_types
    
    def _calculate_risk_score(self, patterns: Dict) -> float:
        """Calculate fraud risk score (0-1)"""
        score = 0.0
        
        # Duplicate items = high risk
        if patterns['duplicate_items'] > 0:
            score += 0.3
        
        # Too many items = medium risk
        if patterns['item_count'] > 50:
            score += 0.2
        
        # High quantity items = medium risk
        if patterns['high_quantity_items'] > 0:
            score += 0.2
        
        # Unusual combinations = high risk
        if patterns['unusual_combinations'] > 0:
            score += 0.3
        
        return min(score, 1.0)
    
    def generate_fraud_report(self, items: List[Dict], total: float, bill_hash: str = None) -> Dict:
        """Generate comprehensive fraud detection report"""
        
        try:
            # Run all checks
            amount_check = self.detect_amount_anomalies(items, total)
            reconciliation_check = self.check_reconciliation(items, total)
            duplicate_check = self.detect_duplicate_bills(items, bill_hash or '')
            pattern_check = self.analyze_item_patterns(items)
            
            # Calculate overall risk
            risk_factors = 0
            if amount_check.get('has_anomalies'):
                risk_factors += 1
            if not reconciliation_check.get('is_reconciled'):
                risk_factors += 1
            if duplicate_check.get('is_duplicate'):
                risk_factors += 2  # Duplicates are critical
            
            overall_risk = min(0.25 * risk_factors + pattern_check['risk_score'] * 0.25, 1.0)
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'overall_risk_score': round(overall_risk, 2),
                'risk_level': 'critical' if overall_risk > 0.8 else 'high' if overall_risk > 0.6 else 'medium' if overall_risk > 0.4 else 'low',
                'checks': {
                    'amount_anomalies': amount_check,
                    'reconciliation': reconciliation_check,
                    'duplicate_detection': duplicate_check,
                    'pattern_analysis': pattern_check
                },
                'flags': self._generate_flags(amount_check, reconciliation_check, duplicate_check, pattern_check),
                'recommendation': self._generate_recommendation(overall_risk)
            }
            
            self.logger.info(f"Fraud report generated. Risk level: {report['risk_level']}")
            return report
            
        except Exception as e:
            self.logger.error(f"Fraud report generation failed: {e}")
            return {'overall_risk_score': 0.0, 'checks': {}, 'flags': []}
    
    def _generate_flags(self, *checks) -> List[str]:
        """Generate human-readable flags"""
        flags = []
        
        for check in checks:
            if isinstance(check, dict):
                if check.get('has_anomalies'):
                    flags.append("⚠️ Amount anomalies detected")
                if not check.get('is_reconciled'):
                    flags.append("⚠️ Bill does not reconcile")
                if check.get('is_duplicate'):
                    flags.append("🚨 Duplicate bill detected")
                if check.get('risk_level') == 'high':
                    flags.append(f"⚠️ High risk pattern detected")
        
        return flags
    
    def _generate_recommendation(self, risk_score: float) -> str:
        """Generate recommendation based on risk"""
        if risk_score > 0.8:
            return "🚨 REJECT - Critical fraud indicators"
        elif risk_score > 0.6:
            return "⚠️ REVIEW - Significant fraud risk, requires manual review"
        elif risk_score > 0.4:
            return "✓ CAUTION - Monitor this bill, proceed with review"
        else:
            return "✓ ACCEPT - Low fraud risk"
