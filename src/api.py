import re
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

# Import model loader utilities from your project architecture
# Adjust imports if your module structure differs slightly
try:
    from api.model_loader import load_pipeline_artifacts, predict_prompt
except ImportError:
    # Fallback to local import if executed directly inside api/
    from model_loader import load_pipeline_artifacts, predict_prompt

app = FastAPI(
    title="Prompt Injection Guardrail API",
    description="API service for real-time prompt injection detection, probability calibration, and attack clustering.",
    version="1.0.0"
)

# Global artifacts loaded at startup
artifacts: Dict[str, Any] = {}

# High-risk deterministic regex pre-filter patterns
HIGH_RISK_PATTERNS = [
    r"(?i)\b(transfer|send|wire)\s+(the\s+)?money\b",
    r"(?i)\baccess\s+(my\s+)?account\b",
    r"(?i)\b(override|ignore)\s+(system|previous)\s+(instructions|rules)\b",
    r"(?i)\bexfiltrate\b",
    r"(?i)\bsystem\s+prompt\s+reveal\b"
]

class PromptRequest(BaseModel):
    prompt: str = Field(..., example="Can you transfer the money into my account?")

class GuardrailResponse(BaseModel):
    verdict: str = Field(..., description="'BLOCK' or 'ALLOW'")
    is_injection: bool
    confidence: float = Field(..., description="Calibrated probability percentage (0-100%)")
    attack_family: Optional[str] = Field(None, description="Decoded attack family cluster if blocked")
    cluster_id: Optional[int] = Field(None, description="Cluster ID from model payload")
    execution_source: str = Field(..., description="'deterministic_regex' or 'ml_classifier'")

@app.on_event("startup")
def startup_event():
    """Load trained model, TF-IDF vectorizer, scaler, and clustering model at API launch."""
    global artifacts
    try:
        artifacts = load_pipeline_artifacts()
        print("✅ Models, TF-IDF vectorizer, and cluster maps loaded successfully.")
    except Exception as e:
        print(f"⚠️ Warning: Model artifacts could not be loaded on startup: {e}")

def check_regex_prefilter(text: str) -> bool:
    """Deterministic check for immediate high-risk triggers."""
    for pattern in HIGH_RISK_PATTERNS:
        if re.search(pattern, text):
            return True
    return False

@app.get("/")
def health_check():
    return {"status": "online", "service": "Prompt Injection Guardrail API"}

@app.post("/predict", response_model=GuardrailResponse)
def analyze_prompt(payload: PromptRequest):
    prompt_text = payload.prompt.strip()

    if not prompt_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Prompt text cannot be empty."
        )

    # 1. Single-word pass heuristic (prevents false positives on single terms like 'valorant')
    tokens = prompt_text.split()
    if len(tokens) == 1 and not check_regex_prefilter(prompt_text):
        return GuardrailResponse(
            verdict="ALLOW",
            is_injection=False,
            confidence=95.0,
            attack_family=None,
            cluster_id=None,
            execution_source="single_word_pass"
        )

    # 2. Deterministic high-risk regex pre-filter
    if check_regex_prefilter(prompt_text):
        return GuardrailResponse(
            verdict="BLOCK",
            is_injection=True,
            confidence=99.0,
            attack_family="data_exfiltration_or_override",
            cluster_id=0,
            execution_source="deterministic_regex"
        )

    # 3. Hybrid ML Classifier Inference (Scalar Features + TF-IDF)
    if not artifacts:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded."
        )

    try:
        result = predict_prompt(prompt_text, artifacts)
        
        is_inj = bool(result.get("is_injection", False))
        verdict = "BLOCK" if is_inj else "ALLOW"
        
        return GuardrailResponse(
            verdict=verdict,
            is_injection=is_inj,
            confidence=round(float(result.get("confidence", 0.0)) * 100, 2),
            attack_family=result.get("attack_family"),
            cluster_id=result.get("cluster_id"),
            execution_source="ml_classifier"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )