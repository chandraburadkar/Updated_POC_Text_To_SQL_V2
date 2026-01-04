import json
import sys
import requests

MCP_BASE = "http://127.0.0.1:9000"


def post(payload: dict):
    url = f"{MCP_BASE}/bridge/run_sql"
    print(f"\n➡️ POST {url}")
    print("Payload:", json.dumps(payload, indent=2))
    r = requests.post(url, json=payload, timeout=30)
    print("Status:", r.status_code)
    try:
        print("Response:", json.dumps(r.json(), indent=2))
    except Exception:
        print("Raw response:", r.text)
    return r.status_code


def main():
    # Try the most common variants to detect key mismatch
    tests = [
        {"sql": "select 1 as x", "limit_preview": 5},
        {"query": "select 1 as x", "limit_preview": 5},
        {"sql": "select 1 as x", "limit": 5},
        {"query": "select 1 as x", "limit": 5},
        # This one should fail (missing sql/query)
        {"final_sql": "select 1 as x", "limit_preview": 5},
    ]

    ok_any = False
    for payload in tests:
        code = post(payload)
        if code == 200:
            ok_any = True

    if not ok_any:
        print("\n❌ MCP /bridge/run_sql did not accept any payload shape.")
        print("Likely server expects different keys OR route not implemented correctly.")
        sys.exit(1)

    print("\n✅ MCP /bridge/run_sql accepted at least one payload shape.")
    sys.exit(0)


if __name__ == "__main__":
    main()