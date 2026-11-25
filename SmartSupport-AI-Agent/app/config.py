import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DB = os.getenv("DATABASE_PATH", str(DATA / "support.db"))
KB = DATA / "knowledge_base.txt"
CSV = DATA / "intent_examples.csv"
