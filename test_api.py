import requests
import json
import time

# Configuration
API_BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("Testing /health endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_extract(file_path):
    """Test extract endpoint"""
    print("\n" + "="*60)
    print(f"Testing /extract endpoint with: {file_path}")
    print("="*60)
    
    try:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return False
        
        with open(file_path, 'rb') as f:
            files = {'document': f}
            
            print("Sending request to API...")
            start_time = time.time()
            
            response = requests.post(
                f"{API_BASE_URL}/extract",
                files=files,
                timeout=60
            )
            
            elapsed = time.time() - start_time
            
            print(f"Status Code: {response.status_code}")
            print(f"Response Time: {elapsed:.2f}s")
            
            result = response.json()
            
            if response.status_code == 200:
                print(f"\n✅ SUCCESS")
                print(f"  Items extracted: {result['data']['total_item_count']}")
                print(f"  Total amount: {result['data']['reconciled_amount']}")
                
                if 'metadata' in result:
                    print(f"  Processing time: {result['metadata']['processing_time_ms']}ms")
                
                return True
            else:
                print(f"\n❌ FAILED")
                print(f"  Error: {result.get('error', 'Unknown error')}")
                return False
    
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == '__main__':
    import os
    
    print("Medical Bill OCR API - Test Suite")
    print("=================================\n")
    
    # Test 1: Health check
    health_ok = test_health()
    
    if not health_ok:
        print("\n⚠️  API is not running!")
        print("Please start the API with: python api.py")
        exit(1)
    
    # Test 2: Extract on samples
    samples = [
        'data/raw/Sample-Document-1.pdf',
        'data/raw/Sample-Document-2.pdf',
        'data/raw/Sample-Document-3.pdf',
        'data/raw/train_sample_4.pdf'
    ]
    
    results = {}
    for sample in samples:
        if os.path.exists(sample):
            results[sample] = test_extract(sample)
        else:
            print(f"\n⚠️  Sample not found: {sample}")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for sample, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{sample}: {status}")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\nTotal: {passed}/{total} tests passed")
