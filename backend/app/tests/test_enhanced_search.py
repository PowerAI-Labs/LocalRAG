import requests
import json
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8000"

def test_basic_search():
    """Test basic enhanced search functionality"""
    query = {
        "question": "What are the main topics?",
        "semantic_search": True,
        "fuzzy_matching": True,
        "page_size": 5
    }
    
    response = requests.post(f"{BASE_URL}/enhanced-search", json=query)
    print("\nBasic Search Test:")
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_filtered_search():
    """Test search with filters"""
    query = {
        "question": "What are the main topics?",
        "filters": {
            "document_types": ["pdf", "docx"],
            "date_range": {
                "start": (datetime.now() - timedelta(days=7)).isoformat(),
                "end": datetime.now().isoformat()
            },
            "min_relevance": 0.5
        },
        "semantic_search": True,
        "page": 1,
        "page_size": 5
    }
    
    response = requests.post(f"{BASE_URL}/enhanced-search", json=query)
    print("\nFiltered Search Test:")
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_sorted_search():
    """Test search with sorting"""
    query = {
        "question": "What are the main topics?",
        "sort": {
            "field": "score",
            "order": "desc"
        },
        "semantic_search": True,
        "page_size": 5
    }
    
    response = requests.post(f"{BASE_URL}/enhanced-search", json=query)
    print("\nSorted Search Test:")
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    # Test with no documents first
    test_basic_search()
    
    # Upload a test document
    with open("sample.pdf", "rb") as f:
        files = {"file": ("sample.pdf", f, "application/pdf")}
        response = requests.post(f"{BASE_URL}/upload", files=files)
        print("\nDocument Upload Test:")
        print(f"Status Code: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
    
    # Run tests after document upload
    test_basic_search()
    test_filtered_search()
    test_sorted_search()