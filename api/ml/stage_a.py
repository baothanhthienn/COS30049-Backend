import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from api import model_loader


def predict_binary(text: str) -> dict:
    model_loader.ensure_loaded()
    return model_loader.predict(text)
