import csv, re, sqlite3
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from app.config import DB, KB, CSV

def init_db():
    Path(DB).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB) as c:
        c.execute("CREATE TABLE IF NOT EXISTS tickets (id INTEGER PRIMARY KEY AUTOINCREMENT, issue TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'open')")

def make_model():
    with open(CSV, newline="", encoding="utf-8") as f: rows = list(csv.DictReader(f))
    model = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), stop_words="english")),
                      ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    model.fit([r["text"] for r in rows], [r["intent"] for r in rows])
    return model

def search_kb(query):
    paragraphs = [p.strip() for p in KB.read_text(encoding="utf-8").split("\n\n") if p.strip()]
    v = TfidfVectorizer(stop_words="english")
    matrix = v.fit_transform(paragraphs + [query])
    scores = (matrix[-1] @ matrix[:-1].T).toarray().ravel()
    return [paragraphs[i] for i in scores.argsort()[::-1][:2] if scores[i] > 0]

class Agent:
    def __init__(self):
        init_db(); self.model = make_model()
    def run(self, message):
        intent = str(self.model.predict([message])[0]); calls = []
        if intent == "create_ticket":
            with sqlite3.connect(DB) as c:
                cur = c.execute("INSERT INTO tickets(issue) VALUES (?)", (message,))
                tid = cur.lastrowid
            calls.append({"tool":"create_ticket","ticket_id":tid,"status":"open"})
            reply = f"Created support ticket #{tid}. Status: open."
        elif intent == "ticket_status":
            match = re.search(r"\b(\d+)\b", message)
            if not match: reply = "Please provide the ticket number."
            else:
                with sqlite3.connect(DB) as c:
                    row = c.execute("SELECT id, issue, status FROM tickets WHERE id=?", (int(match.group(1)),)).fetchone()
                calls.append({"tool":"get_ticket_status","result":row})
                reply = f"Ticket details: {row}" if row else "Ticket not found."
        elif intent == "knowledge":
            hits = search_kb(message); calls.append({"tool":"search_knowledge_base","matches":len(hits)})
            reply = "\n\n".join(hits) if hits else "No matching help article found. You can request a support ticket."
        else:
            reply = "Please describe your account, password, order, or support issue in more detail."
        return {"reply":reply, "intent":intent, "tool_calls":calls}
