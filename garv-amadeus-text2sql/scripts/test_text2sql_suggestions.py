import requests, json

url = "http://127.0.0.1:8000/api/text2sql"
payload = {
    "question": "Top 5 airports by avg security wait time in last 30 days",
    "top_k_schema": 5,
    "return_rows": 20,
    "enable_viz": True,
}
r = requests.post(url, json=payload, timeout=120)
print("Status:", r.status_code)
out = r.json()
print("suggested_questions:", out.get("suggested_questions"))
print("trace last:", (out.get("debug") or {}).get("trace", [])[-1])