from paddleocr import PaddleOCR
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class OCREngine:
    """Paddle OCR wrapper"""
    
    def __init__(self):
        logger.info("Initializing Paddle OCR...")
        try:
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
            logger.info("Paddle OCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Paddle OCR: {e}")
            self.ocr = None
    
    def extract_text(self, image_path):
        """Extract text from image"""
        
        if self.ocr is None:
            logger.error("OCR not initialized")
            return {
                'text_blocks': [],
                'full_text': '',
                'lines': []
            }
        
        try:
            # Check if file exists
            if not image_path or not isinstance(image_path, str):
                logger.error("Invalid image path")
                return {'text_blocks': [], 'full_text': '', 'lines': []}
            
            # Read image to verify
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Could not read image: {image_path}")
                return {'text_blocks': [], 'full_text': '', 'lines': []}
            
            logger.info(f"Running OCR on: {image_path}")
            
            # Run OCR
            result = self.ocr.ocr(image_path, cls=True)
            
            if not result or not result:
                logger.warning("OCR returned no results")
                return {
                    'text_blocks': [],
                    'full_text': '',
                    'lines': []
                }
            
            # Parse results
            text_blocks = []
            
            for line in result:
                if not line:
                    continue
                
                for word_info in line:
                    try:
                        bbox, (text, confidence) = word_info
                        
                        text_blocks.append({
                            'text': text,
                            'confidence': float(confidence),
                            'bbox': {
                                'x_min': float(bbox),
                                'y_min': float(bbox),
                                'x_max': float(bbox),
                                'y_max': float(bbox)
                            }
                        })
                    except Exception as e:
                        logger.debug(f"Error parsing word: {e}")
                        continue
            
            logger.info(f"Extracted {len(text_blocks)} text blocks")
            
            # Group by lines
            lines = OCREngine._group_by_line(text_blocks)
            
            # Reconstruct full text
            full_text = ' '.join([block['text'] for block in text_blocks])
            
            return {
                'text_blocks': text_blocks,
                'full_text': full_text,
                'lines': lines
            }
        
        except Exception as e:
            logger.error(f"OCR error: {e}", exc_info=True)
            return {
                'text_blocks': [],
                'full_text': '',
                'lines': []
            }
    
    @staticmethod
    def _group_by_line(text_blocks):
        """Group text blocks by y-coordinate"""
        
        if not text_blocks:
            return []
        
        lines = []
        sorted_blocks = sorted(text_blocks, key=lambda b: b['bbox']['y_min'])
        
        current_line = []
        current_y = None
        
        for block in sorted_blocks:
            y = block['bbox']['y_min']
            
            # If y-coordinate is close, consider it same line
            if current_y is None or abs(y - current_y) < 15:
                current_line.append(block)
                if current_y is None:
                    current_y = y
            else:
                if current_line:
                    lines.append(current_line)
                current_line = [block]
                current_y = y
        
        if current_line:
            lines.append(current_line)
        
        return lines
