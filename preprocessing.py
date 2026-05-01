import cv2
import numpy as np
import fitz  # PyMuPDF
import logging
import os

logger = logging.getLogger(__name__)

class Preprocessor:
    """Image preprocessing for OCR"""
    
    @staticmethod
    def process_image(image_path):
        """Simple 3-step preprocessing"""
        
        logger.info(f"Processing: {image_path}")
        
        # Step 1: PDF to Image
        if image_path.endswith('.pdf'):
            images = Preprocessor.pdf_to_images(image_path)
        else:
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"Could not read: {image_path}")
                return []
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images = [gray]
        
        logger.info(f"Converted to {len(images)} images")
        
        processed = []
        for idx, img in enumerate(images):
            try:
                # Step 2: Enhance contrast with CLAHE
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                img = clahe.apply(img)
                
                # Step 3: Binarize (adaptive threshold)
                img = cv2.adaptiveThreshold(
                    img, 255, 
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 
                    11, 2
                )
                
                processed.append(img)
                logger.info(f"Processed page {idx+1}")
                
            except Exception as e:
                logger.error(f"Error processing page {idx}: {e}")
                processed.append(img)
        
        return processed
    
    @staticmethod
    def pdf_to_images(pdf_path, dpi=300):
        """Convert PDF to images at 300 DPI"""
        
        try:
            pdf = fitz.open(pdf_path)
            images = []
            
            for page_num in range(len(pdf)):
                try:
                    page = pdf[page_num]
                    zoom = dpi / 72
                    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
                    
                    # Convert to numpy array
                    img_data = pix.samples
                    img_array = np.frombuffer(img_data, dtype=np.uint8)
                    img_array = img_array.reshape(pix.h, pix.w, pix.n)
                    
                    # Convert RGBA to grayscale if needed
                    if pix.n == 4:  # RGBA
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
                    elif pix.n == 3:  # RGB
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    
                    images.append(img_array)
                    logger.info(f"Converted page {page_num + 1}")
                    
                except Exception as e:
                    logger.error(f"Error converting page {page_num}: {e}")
                    continue
            
            pdf.close()
            return images
        
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            return []
