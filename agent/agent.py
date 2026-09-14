import sys
import requests

API_URL = "http://127.0.0.1:8000/predict"

def main():
    print("=" * 65)
    print("      🛡️  PROMPT INJECTION GUARDRAIL AGENT ACTIVE  🛡️")
    print("   Type your message and press Enter. Type 'exit' to quit.")
    print("=" * 65 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nAgent: Goodbye! Shutting down session.")
                break

            # Prevent empty inputs from triggering Pydantic min_length=1 validation error
            if not user_input:
                print("🤖 Agent [WARNING]: Prompt cannot be empty. Please enter text.\n")
                continue

            # Check character limit to respect min_length=1 and max_length=10000
            if len(user_input) > 10000:
                print("🤖 Agent [WARNING]: Text exceeds maximum limit of 10,000 characters.\n")
                continue

            # Send payload structured to match PredictRequest schema
            payload = {"text": user_input}
            headers = {"Content-Type": "application/json"}

            response = requests.post(API_URL, json=payload, headers=headers)

            if response.status_code == 200:
                data = response.json()
                verdict = data.get("verdict", "UNKNOWN")
                confidence = data.get("confidence", 0.0)
                cluster_label = data.get("cluster_label")

                if verdict == "BLOCK":
                    print(f"\n🤖 Agent [GUARDRAIL BLOCKED]: 🚨 Prompt injection detected!")
                    print(f"   ├── Verdict: {verdict} (Label: {data.get('label')})")
                    print(f"   ├── Confidence: {confidence * 100:.2f}%")
                    if cluster_label:
                        print(f"   ├── Attack Family: {cluster_label} (Cluster ID: {data.get('cluster_id')})")
                    if data.get("spans"):
                        print(f"   ├── Malicious Spans Detected: {data.get('spans')}")
                    print(f"   └── Decoded Text: \"{data.get('decoded_text')}\"\n")
                else:
                    print(f"\n🤖 Agent [ALLOWED]: Prompt verified safe.")
                    print(f"   ├── Confidence: {(1 - confidence) * 100:.2f}% safe")
                    print(f"   └── Decoded Text: \"{data.get('decoded_text')}\"\n")

            elif response.status_code == 422:
                print(f"\n🤖 Agent [422 Validation Error]: {response.json().get('detail')}\n")
            else:
                print(f"\n🤖 Agent [ERROR]: Server returned status code {response.status_code}.\n")

        except requests.exceptions.ConnectionError:
            print("\n🤖 Agent [ERROR]: Cannot connect to FastAPI server at http://127.0.0.1:8000. Is Uvicorn running?\n")
            break
        except KeyboardInterrupt:
            print("\n\nAgent: Session interrupted. Exiting...")
            sys.exit(0)

if __name__ == "__main__":
    main()