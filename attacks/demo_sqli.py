import argparse
import sys

import httpx

SQLI_PAYLOAD = "' OR '1'='1 --"


def main(url: str, session_id: str) -> None:
    print(f"SQLi probe against {url}/api/sessions/{session_id}/history")
    print(f"  payload: {SQLI_PAYLOAD!r}\n")

    with httpx.Client(timeout=15) as client:
        base = client.get(f"{url}/api/sessions/{session_id}/history", params={"limit": 1})
        if base.status_code == 404:
            print("Session not found — create a few classifications first.")
            sys.exit(1)
        if base.status_code != 200:
            print(f"Unexpected status {base.status_code}: {base.text}")
            sys.exit(1)
        total_without_filter = base.json()["total"]

        # Attack: inject payload via q=
        attack = client.get(
            f"{url}/api/sessions/{session_id}/history",
            params={"limit": 1, "q": SQLI_PAYLOAD},
        )

    print(f"  Records without filter : {total_without_filter}")
    print(f"  Status with SQLi payload: {attack.status_code}")

    if attack.status_code == 500:
        print("\n✗ FAIL — server returned 500; raw SQL error may have leaked")
        sys.exit(1)

    if attack.status_code != 200:
        print(f"\n✗ FAIL — unexpected status {attack.status_code}")
        sys.exit(1)

    total_with_payload = attack.json()["total"]
    print(f"  Records with SQLi payload: {total_with_payload}")

    if total_with_payload == total_without_filter and total_without_filter > 0:
        print("\n✗ FAIL — payload returned all rows; SQLi may be effective")
        sys.exit(1)

    print("\n✓ PASS — payload treated as literal string, not executed as SQL")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()
    main(args.url, args.session_id)
