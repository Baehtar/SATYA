"""Judging benchmark evaluation set for Satya API."""
import sys
import time
import httpx

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

API_BASE = "http://localhost:8000"

TEST_CASES = [
    {
        "id": "tc1_kerala_flood",
        "type": "text",
        "payload": {"text": "This photo showing cyclone damage yesterday in southern state!"},
        "expected_verdict": ["likely_false", "unverifiable"]
    },
    {
        "id": "tc2_unesco_anthem",
        "type": "text",
        "payload": {"text": "UNESCO has officially declared Jana Gana Mana as the best national anthem in the world!"},
        "expected_verdict": ["likely_false"]
    },
    {
        "id": "tc3_rbi_nano_chip",
        "type": "text",
        "payload": {"text": "Breaking: RBI 2000 rupee notes contain a secret satellite GPS nano chip embedded inside!"},
        "expected_verdict": ["likely_false"]
    },
    {
        "id": "tc4_modi_15_lakh",
        "type": "text",
        "payload": {"text": "Government is depositing 15 lakh rupees in everyone bank account tomorrow under PM scheme!"},
        "expected_verdict": ["likely_false"]
    },
    {
        "id": "tc5_eci_election",
        "type": "text",
        "payload": {"text": "Election Commission of India conducts free and fair elections for parliament."},
        "expected_verdict": ["likely_true", "unverifiable"]
    },
    {
        "id": "tc6_unverifiable_greeting",
        "type": "text",
        "payload": {"text": "Good morning my dear friends have a great Monday ahead!"},
        "expected_verdict": ["unverifiable"]
    }
]


def run_benchmark():
    print("=" * 60)
    print("      SATYA API EVALUATION BENCHMARK SUITE")
    print("=" * 60)

    # 1. Health check
    try:
        r = httpx.get(f"{API_BASE}/api/health", timeout=5.0)
        if r.status_code != 200 or r.json().get("status") != "ok":
            print(f"[FAIL] Health check failed: {r.text}")
            sys.exit(1)
        print("[OK] Server Health Check: OK")
    except Exception as e:
        print(f"[FAIL] Server connection failed: {e}")
        print("Please start the server first using `python run.py`!")
        sys.exit(1)

    passed = 0
    total = len(TEST_CASES)

    for case in TEST_CASES:
        case_id = case["id"]
        print(f"\nRunning {case_id} ({case['type']})...")
        t0 = time.time()
        
        # Submit check
        resp = httpx.post(f"{API_BASE}/api/check", data=case["payload"], timeout=10.0)
        if resp.status_code != 200:
            print(f"  ❌ Failed submission HTTP {resp.status_code}")
            continue
            
        check_id = resp.json().get("id")
        
        # Stream SSE results
        verdict_received = None
        with httpx.stream("GET", f"{API_BASE}/api/check/{check_id}/stream", timeout=30.0) as stream:
            for line in stream.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if '"verdict":' in data_str:
                        import json
                        try:
                            verdict_received = json.loads(data_str)
                        except Exception:
                            pass
                            
        latency = (time.time() - t0) * 1000
        
        if verdict_received and "verdict" in verdict_received:
            v_val = verdict_received["verdict"]
            conf = verdict_received.get("confidence", 0)
            en_exp = verdict_received.get("explanation_en", "")[:70]
            print(f"  Verdict: {v_val.upper()} | Confidence: {round(conf*100)}% | Latency: {round(latency)}ms")
            print(f"  En: {en_exp}...")
            
            if v_val in case["expected_verdict"]:
                print("  [PASS] OK")
                passed += 1
            else:
                print(f"  [WARN] Match difference (Got {v_val}, expected {case['expected_verdict']})")
        else:
            print("  [FAIL] No verdict returned")

    print("\n" + "=" * 60)
    print(f"BENCHMARK SUMMARY: {passed}/{total} Passed ({round(passed/total*100)}%)")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
