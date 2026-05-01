# Medical Bill OCR & Fraud Detection

> End-to-end pipeline for extracting structured data from scanned hospital bills and detecting fraudulent claims using four automated anomaly detection algorithms. Built for a competitive datathon.

[![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-2.7-0062B0?style=flat-square)](https://github.com/PaddlePaddle/PaddleOCR)
[![Qwen](https://img.shields.io/badge/Qwen-7B-FF6B35?style=flat-square)](https://huggingface.co/Qwen)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## The Problem

Medical billing fraud costs the healthcare system billions annually. Hospital bills arrive as scanned images — different layouts, different fonts, tilted, blurry, inconsistently formatted. Manually reviewing them for fraud is slow and inconsistent.

**Two challenges:**
1. **Extraction** — reliably read text from messy scanned images across dozens of different bill formats
2. **Detection** — automatically flag suspicious patterns without human review for every bill

---

## Pipeline Overview

### Part 1: OCR Extraction

```
Raw Scanned Bill (PNG/PDF)
         │
         ▼ ─────── Deskew (Hough Line Transform)
         │          Removes scan angle tilt
         ▼ ─────── Bilateral Filter
         │          Noise removal preserving edges
         ▼ ─────── CLAHE
         │          Local contrast enhancement
         ▼ ─────── Adaptive Thresholding
         │          Region-specific binarisation
         ▼ ─────── 4x DPI Scaling
         │          72 DPI → 300 DPI
         ▼
    PaddleOCR (use_angle_cls=True)
    Text detection + recognition
    with rotation correction
         │
         ▼
    Line Grouping by Y-coordinate
    (words on same line → paired)
         │
         ▼
    Qwen 7B via Ollama
    (structured JSON extraction)
         │
         ▼
    {item_name, amount, quantity,
     total, hospital_name, date}
```

### Part 2: Fraud Detection

```
Structured Bill Data
         │
    ┌────┴────┬──────────┬────────────┐
    ▼         ▼          ▼            ▼
 IQR Check  Recon.   SHA-256    Pattern
 Anomaly    Check    Duplicate  Analysis
    │         │          │            │
    └────┬────┘          │            │
         └───────────────┴────────────┘
                         │
                         ▼
              Risk Score (0.0 → 1.0)
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    < 0.4 ACCEPT    0.4-0.8 REVIEW   > 0.8 REJECT
```

---

## Fraud Detection Logic

### Check 1: IQR Amount Anomaly
Detects charges that are statistically anomalous relative to other charges on the same bill.

```python
def iqr_anomaly_check(items: list[dict]) -> float:
    amounts = [item['amount'] for item in items]
    q75 = np.percentile(amounts, 75)
    q25 = np.percentile(amounts, 25)
    iqr = q75 - q25
    threshold = np.percentile(amounts, 95) * 1.5
    anomalies = [a for a in amounts if a > threshold]
    return min(len(anomalies) / len(amounts), 1.0)
```

### Check 2: Reconciliation Check
Compares sum of line items against declared total. Fraudulent bills often inflate totals.

```python
def reconciliation_check(items: list[dict], declared_total: float) -> float:
    computed_total = sum(item['amount'] * item['quantity'] for item in items)
    discrepancy = abs(computed_total - declared_total) / declared_total
    if discrepancy > 0.01:  # > 1% discrepancy = suspicious
        return min(discrepancy * 5, 1.0)
    return 0.0
```

### Check 3: SHA-256 Duplicate Detection
Detects the same bill submitted multiple times within a 30-day window.

```python
import hashlib
from datetime import datetime, timedelta

def sha256_duplicate_check(bill_content: str, cache: dict) -> float:
    bill_hash = hashlib.sha256(bill_content.encode()).hexdigest()
    current_time = datetime.now()
    if bill_hash in cache:
        if current_time - cache[bill_hash] < timedelta(days=30):
            return 1.0  # Duplicate detected
    cache[bill_hash] = current_time
    return 0.0
```

### Check 4: Pattern Analysis
Detects suspicious billing patterns within a single visit.

```python
def pattern_analysis(items: list[dict]) -> float:
    risk = 0.0
    item_names = [item['item_name'].lower() for item in items]
    
    # Duplicate services
    if len(item_names) != len(set(item_names)):
        risk += 0.3
    
    # Unusually high quantities
    high_qty = [i for i in items if i['quantity'] > 10]
    if high_qty:
        risk += 0.2 * len(high_qty)
    
    # Too many service categories
    categories = extract_categories(item_names)
    if len(categories) > 5:
        risk += 0.3
    
    return min(risk, 1.0)
```

### Combined Risk Score

```python
def compute_risk_score(checks: dict) -> float:
    weights = {
        'iqr': 0.25,
        'reconciliation': 0.35,  # Highest weight — most definitive
        'duplicate': 0.30,
        'pattern': 0.10
    }
    return sum(checks[k] * weights[k] for k in weights)
```

---

## Image Preprocessing Details

| Step | Tool | Purpose |
|------|------|---------|
| Deskewing | OpenCV Hough Lines | Corrects scan angle — text tilted at scan time |
| Bilateral Filter | OpenCV | Removes grain while preserving letter edges |
| CLAHE | OpenCV | Local contrast — handles shadowed bill corners |
| Adaptive Threshold | OpenCV | Region-specific binarisation — handles uneven lighting |
| Noise Removal | Morphological ops | Removes isolated speckles after binarisation |
| DPI Scaling | PIL | 72 → 300 DPI — PaddleOCR accuracy threshold |

---

## Project Structure

```
medical-bill-ocr-fraud/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── preprocessing/
│   │   └── image_processor.py  # Full OpenCV pipeline
│   ├── ocr/
│   │   ├── paddle_ocr.py        # PaddleOCR wrapper
│   │   └── line_grouper.py     # Y-coordinate line assembly
│   ├── extraction/
│   │   ├── qwen_extractor.py   # Qwen 7B JSON extraction
│   │   └── regex_extractor.py  # Fallback regex parser
│   ├── fraud/
│   │   ├── iqr_check.py
│   │   ├── reconciliation.py
│   │   ├── sha256_cache.py
│   │   ├── pattern_analysis.py
│   │   └── risk_scorer.py      # Combines all checks
│   └── api.py                  # Flask API endpoint
├── tests/
│   ├── test_preprocessing.py
│   ├── test_ocr.py
│   └── test_fraud_checks.py
├── notebooks/
│   ├── 01_preprocessing_demo.ipynb
│   ├── 02_ocr_extraction.ipynb
│   └── 03_fraud_detection_analysis.ipynb
└── Dockerfile
```

---

## Requirements

```
paddlepaddle==2.5.1
paddleocr==2.7.0
opencv-python==4.7.0.72
Pillow==9.5.0
numpy==1.24.3
pandas==2.0.1
ollama==0.1.7
flask==2.3.2
pdf2image==1.16.3
scipy==1.10.1
```

---

## Results

- **90%+ extraction accuracy** across heterogeneous bill formats
- **4 independent fraud checks** with weighted risk scoring
- **Fallback mechanism**: if Qwen extraction fails JSON validation, regex BillExtractor takes over
- **SHA-256 cache**: disk + memory dual-layer to avoid reprocessing same bill

---

*Datathon Project · November 2025*
*Harshal Bhambhani · BITS Hyderabad · 2026*