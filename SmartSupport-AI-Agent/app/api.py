from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.agent import Agent
app = FastAPI(title="SmartSupport AI Agent API")
agent = Agent()
class Chat(BaseModel):
    message: str = Field(min_length=2, max_length=2000)
@app.get("/health")
def health(): return {"status":"ok"}
@app.post("/chat")
def chat(req: Chat): return agent.run(req.message)
