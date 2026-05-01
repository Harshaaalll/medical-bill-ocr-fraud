from .preprocessing import Preprocessor
from .ocr_engine import OCREngine
from .extraction import BillExtractor
import logging
import time
import cv2
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalBillPipeline:
    """Complete extraction pipeline"""
    
    def __init__(self):
        logger.info("Initializing pipeline...")
        self.preprocessor = Preprocessor()
        self.ocr_engine = OCREngine()
        self.extractor = BillExtractor()
        
        # Create temp directory
        os.makedirs('/tmp/medical_bill_ocr', exist_ok=True)
        logger.info("Pipeline initialized")
    
    def process(self, file_path):
        """Process bill and extract data"""
        
        start_time = time.time()
        logger.info(f"Processing: {file_path}")
        
        try:
            # Validate file
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return self._error_response("File not found")
            
            # Step 1: Preprocessing
            logger.info("Step 1: Preprocessing...")
            processed_images = self.preprocessor.process_image(file_path)
            
            if not processed_images:
                logger.error("No images after preprocessing")
                return self._error_response("No images processed")
            
            logger.info(f"Got {len(processed_images)} processed images")
            
            # Step 2: OCR on first page only (for now)
            logger.info("Step 2: OCR extraction...")
            
            # Save first processed image temporarily
            temp_path = '/tmp/medical_bill_ocr/temp_ocr.png'
            cv2.imwrite(temp_path, processed_images)
            
            if not os.path.exists(temp_path):
                logger.error("Failed to save temp image")
                return self._error_response("Failed to save temp image")
            
            ocr_result = self.ocr_engine.extract_text(temp_path)
            ocr_text = ocr_result.get('full_text', '')
            
            logger.info(f"OCR extracted {len(ocr_text)} characters")
            
            # Step 3: Extract items
            logger.info("Step 3: Item extraction...")
            items = self.extractor.extract_items(ocr_text)
            logger.info(f"Extracted {len(items)} items")
            
            # Step 4: Calculate total
            extracted_total = sum(item['item_amount'] for item in items)
            
            # Step 5: Find bill total
            bill_total = self.extractor.find_bill_total(ocr_text)
            
            # Calculate reconciliation score
            if bill_total and bill_total > 0:
                discrepancy = abs(extracted_total - bill_total)
                reconciliation_score = max(0, 1 - (discrepancy / bill_total))
            else:
                reconciliation_score = 0.5  # Neutral if can't find total
            
            processing_time = time.time() - start_time
            
            logger.info(f"Processing completed in {processing_time:.2f}s")
            logger.info(f"Extracted {len(items)} items, Total: {extracted_total}")
            
            return {
                'is_success': True,
                'data': {
                    'pagewise_line_items': [
                        {
                            'page_no': '1',
                            'bill_items': items
                        }
                    ],
                    'total_item_count': len(items),
                    'reconciled_amount': round(extracted_total, 2)
                },
                'metadata': {
                    'processing_time_ms': round(processing_time * 1000, 2),
                    'bill_total_detected': bill_total,
                    'reconciliation_score': round(reconciliation_score, 2)
                }
            }
        
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            return self._error_response(str(e))
    
    @staticmethod
    def _error_response(error_msg):
        return {
            'is_success': False,
            'error': error_msg,
            'data': {
                'pagewise_line_items': [],
                'total_item_count': 0,
                'reconciled_amount': 0.0
            }
        }
