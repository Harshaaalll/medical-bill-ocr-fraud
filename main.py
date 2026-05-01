"""
Main entry point for medical bill OCR system
"""

import os
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils import load_config, ensure_dir, save_json, list_files
from preprocessing import ImagePreprocessor
from ocr_engine import OCREngine
from extraction import FieldExtractor
from validation import AmountValidator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ocr_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def process_single_bill(image_path: str, config: dict) -> dict:
    """
    Process a single medical bill
    
    Args:
        image_path: Path to bill image
        config: Configuration dictionary
        
    Returns:
        Processing results
    """
    logger.info(f"=" * 60)
    logger.info(f"Processing: {image_path}")
    logger.info(f"=" * 60)
    
    # Initialize modules
    preprocessor = ImagePreprocessor(config)
    ocr_engine = OCREngine(config)
    extractor = FieldExtractor(config)
    validator = AmountValidator()
    
    # Get paths
    filename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(filename)[0]
    
    preprocessed_dir = config['paths']['preprocessed']
    outputs_dir = config['paths']['outputs']
    
    ensure_dir(preprocessed_dir)
    ensure_dir(outputs_dir)
    
    # Step 1: Preprocessing
    logger.info("Step 1: Preprocessing")
    preprocessed_path = os.path.join(preprocessed_dir, f"{name_without_ext}_preprocessed.png")
    preprocessed_image = preprocessor.process(image_path, preprocessed_path)
    
    # Step 2: OCR
    logger.info("Step 2: OCR - Text Extraction")
    ocr_data = ocr_engine.extract_text(preprocessed_path)
    
    # Step 3: Field Extraction
    logger.info("Step 3: Field Extraction")
    line_items = extractor.extract_line_items(ocr_data)
    
    # Step 4: Validation
    logger.info("Step 4: Validation")
    validation_result = validator.validate(line_items, ocr_data['full_text'])
    
    # Prepare output
    output = {
        'is_success': True,
        'source_file': filename,
        'data': {
            'pagewise_line_items': [
                {
                    'page_no': '1',
                    'bill_items': [
                        {
                            'item_name': item['item_name'],
                            'item_amount': item['item_amount'],
                            'item_rate': item['item_rate'],
                            'item_quantity': item['item_quantity']
                        }
                        for item in validation_result['line_items']
                    ]
                }
            ],
            'total_item_count': len(validation_result['line_items']),
            'reconciled_amount': validation_result['extracted_total']
        },
        'validation': {
            'is_reconciled': validation_result['is_reconciled'],
            'bill_total': validation_result.get('bill_total'),
            'discrepancy': validation_result.get('discrepancy', 0)
        }
    }
    
    # Save output
    output_path = os.path.join(outputs_dir, f"{name_without_ext}_output.json")
    save_json(output, output_path)
    
    logger.info(f"Processing complete. Output saved to {output_path}")
    logger.info(f"Found {len(line_items)} line items")
    logger.info(f"Total amount: {validation_result['extracted_total']}")
    logger.info(f"Reconciled: {validation_result['is_reconciled']}")
    
    return output


def main():
    """
    Main function - process all bills in raw data folder
    """
    logger.info("Medical Bill OCR System - Starting")
    
    # Load configuration
    config = load_config()
    
    # Get input files
    raw_dir = config['paths']['raw_data']
    image_files = list_files(raw_dir, extension='.png') + \
                  list_files(raw_dir, extension='.jpg') + \
                  list_files(raw_dir, extension='.jpeg')
    
    if len(image_files) == 0:
        logger.error(f"No images found in {raw_dir}")
        logger.info("Please add bill images to data/raw/ folder")
        return
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Process each file
    results = []
    
    for image_path in image_files:
        try:
            result = process_single_bill(image_path, config)
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}", exc_info=True)
            results.append({
                'is_success': False,
                'source_file': os.path.basename(image_path),
                'error': str(e)
            })
    
    # Summary
    successful = sum(1 for r in results if r['is_success'])
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Processing Summary")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total files: {len(image_files)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {len(image_files) - successful}")
    logger.info(f"Check data/outputs/ for results")
    logger.info(f"Check ocr_pipeline.log for detailed logs")


if __name__ == "__main__":
    main()
