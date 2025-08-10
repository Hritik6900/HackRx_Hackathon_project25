import requests
import json
import time
from PyPDF2 import PdfReader
import io

def test_pdf_access():
    """Test PDF access directly"""
    print("\n5. PDF ACCESS TEST:")
    
    pdf_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    try:
        print("   Downloading PDF...")
        response = requests.get(pdf_url, timeout=20)
        print(f"   PDF Status: {response.status_code}")
        print(f"   PDF Size: {len(response.content)} bytes")
        
        if response.status_code == 200:
            pdf_reader = PdfReader(io.BytesIO(response.content))
            print(f"   PDF Pages: {len(pdf_reader.pages)}")
            
            # Extract first page text
            if len(pdf_reader.pages) > 0:
                first_page_text = pdf_reader.pages[0].extract_text()
                print(f"   Text Length: {len(first_page_text)} chars")
                print(f"   Text Sample: {first_page_text[:200]}...")
                
                # Check for key insurance terms
                key_terms = ["grace period", "premium", "policy", "insurance", "mediclaim"]
                found_terms = [term for term in key_terms if term.lower() in first_page_text.lower()]
                print(f"   Key Terms Found: {found_terms}")
                
                if len(first_page_text) > 100 and found_terms:
                    print("   PDF ACCESS: SUCCESS")  # Removed Unicode
                    return True
                else:
                    print("   PDF ACCESS: FAILED (No meaningful content)")
                    return False
            else:
                print("   PDF ACCESS: FAILED (No pages found)")
                return False
        else:
            print(f"   PDF ACCESS: FAILED (Status: {response.status_code})")
            print(f"   Error: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"   PDF ACCESS: ERROR - {e}")  # Removed Unicode
        return False

def test_webhook():
    webhook_url = "https://hackrx-hackathon-project25.onrender.com/hackrx/run"
    
    headers = {
        "Authorization": "Bearer hackrx-api-key-2025",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    payload = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?"
        ]
    }
    
    print("=" * 60)
    print("HACKRX WEBHOOK TEST - COMPLETE DIAGNOSIS")
    print("=" * 60)
    
    # Step 1: Test basic connectivity
    print("\n1. CONNECTIVITY TEST:")
    try:
        base_url = webhook_url.replace("/hackrx/run", "")
        response = requests.get(base_url, timeout=10)
        print(f"   Base URL Status: {response.status_code}")
    except Exception as e:
        print(f"   Base URL Error: {e}")
    
    # Step 2: Test health endpoint
    print("\n2. HEALTH CHECK:")
    try:
        health_url = base_url + "/health"
        response = requests.get(health_url, timeout=10)
        print(f"   Health Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   Health Response: {response.json()}")
        else:
            print(f"   Health Error: {response.text[:200]}")
    except Exception as e:
        print(f"   Health Error: {e}")
    
    # Step 3: Test debug endpoint
    print("\n3. DEBUG INFO:")
    try:
        debug_url = base_url + "/debug"
        response = requests.get(debug_url, timeout=10)
        print(f"   Debug Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   Debug Info: {response.json()}")
        elif response.status_code == 404:
            print("   Debug endpoint not available (normal)")
    except Exception as e:
        print(f"   Debug Not Available: {e}")
    
    # Step 4: PDF Access Test
    pdf_success = test_pdf_access()
    
    # Step 5: Main webhook test
    print("\n6. MAIN WEBHOOK TEST:")
    print(f"   URL: {webhook_url}")
    print(f"   Questions: {len(payload['questions'])}")
    print("-" * 50)
    
    try:
        start_time = time.time()
        
        response = requests.post(
            webhook_url,
            headers=headers,
            json=payload,
            timeout=60
        )
        
        end_time = time.time()
        processing_time = round(end_time - start_time, 2)
        
        print(f"   Processing Time: {processing_time} seconds")
        print(f"   Status Code: {response.status_code}")
        print("-" * 50)
        
        if response.status_code == 200:
            try:
                result = response.json()
                print("   SUCCESS! JSON Response:")
                print(json.dumps(result, indent=4))
                
                # Analyze response quality
                if "answers" in result and isinstance(result["answers"], list):
                    print(f"\n   FORMAT VALIDATION: PASSED")
                    print(f"   Answer Count: {len(result['answers'])}")
                    
                    # Check if answers are generic errors or actual content
                    generic_responses = [
                        "Unable to process this question due to processing constraints",
                        "Processing error occurred",
                        "Unable to extract relevant information"
                    ]
                    
                    actual_answers = 0
                    for i, answer in enumerate(result["answers"]):
                        if isinstance(answer, str) and len(answer) > 10:
                            if not any(generic in answer for generic in generic_responses):
                                actual_answers += 1
                                print(f"   Answer {i+1}: REAL CONTENT ({len(answer)} chars)")
                            else:
                                print(f"   Answer {i+1}: GENERIC ERROR ({len(answer)} chars)")
                        else:
                            print(f"   Answer {i+1}: INVALID")
                    
                    if actual_answers > 0:
                        print(f"\n   CONTENT QUALITY: GOOD ({actual_answers}/3 real answers)")
                        print(f"   WEBHOOK TEST: SUCCESSFUL")
                    else:
                        print(f"\n   CONTENT QUALITY: POOR (0/3 real answers)")
                        print(f"   WEBHOOK TEST: FAILED (Generic responses only)")
                        
                        if not pdf_success:
                            print("   LIKELY CAUSE: PDF access issues")
                        else:
                            print("   LIKELY CAUSE: AI processing issues - Check Render logs")
                    
                    return actual_answers > 0
                else:
                    print(f"\n   FORMAT VALIDATION: FAILED")
                    return False
                    
            except json.JSONDecodeError:
                print(f"   JSON Parse Error. Raw Response:")
                print(f"   {response.text[:500]}")
                return False
                
        else:
            print(f"   HTTP ERROR: {response.status_code}")
            print(f"   Response: {response.text[:300]}")
            return False
            
    except Exception as e:
        print(f"   ERROR: {e}")
        return False

if _name_ == "_main_":
    print("Starting comprehensive webhook test...")
    success = test_webhook()
    
    print("\n" + "=" * 60)
    if success:
        print("TEST RESULT: SUCCESS - API working with real content")
    else:
        print("TEST RESULT: FAILED - API needs debugging")
    print("=" * 60)