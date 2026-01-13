import urllib.request
import urllib.error
import json
import sys
import time

def check_url(url, description):
    print(f"Checking {description} ({url})...")
    try:
        start_time = time.time()
        with urllib.request.urlopen(url, timeout=5) as response:
            content = response.read()
            elapsed = time.time() - start_time
            print(f"SUCCESS: {description} responded in {elapsed:.2f}s")
            return True, content
    except urllib.error.URLError as e:
        print(f"FAILED: {description} - {e}")
        return False, None
    except Exception as e:
        print(f"ERROR: {description} - {e}")
        return False, None

def main():
    print("Starting diagnosis...")
    
    # Check 1: Root URL (HTML)
    ok, html = check_url("http://127.0.0.1:8000/", "Frontend Root")
    if ok:
        print(f"Root content length: {len(html)}")
        if b"Wiki AI" in html:
            print("Root content seems correct.")
        else:
            print("WARNING: Root content might be incorrect.")

    # Check 2: API Endpoint
    ok, data = check_url("http://127.0.0.1:8000/api/v1/folders/0/contents", "API Folders")
    if ok:
        try:
            json_data = json.loads(data)
            print("API returned valid JSON.")
            print(f"Documents count: {len(json_data.get('documents', []))}")
        except json.JSONDecodeError:
            print("FAILED: API did not return valid JSON.")

    # Check 3: Static Vendor File
    check_url("http://127.0.0.1:8000/static/vendor/vue.global.prod.min.js", "Vue JS Library")
    check_url("http://127.0.0.1:8000/static/vendor/tailwindcss.js", "Tailwind JS Library")

if __name__ == "__main__":
    main()
