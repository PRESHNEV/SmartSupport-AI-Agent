from pathlib import Path
import joblib
from app.agent import make_model
out = Path("artifacts/intent_classifier.joblib")
if __name__ == "__main__":
    out.parent.mkdir(exist_ok=True)
    joblib.dump(make_model(), out)
    print("Saved", out)
