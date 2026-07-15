from fastapi import FastAPI

app = FastAPI(title="GenAI QA Mock API")


@app.get("/")
def home():
    return {
        "message": "GenAI QA Mock API is running successfully!"
    }


@app.post("/chat")
def chat(request: dict):
    question = request.get("question", "")

    return {
        "question": question,
        "answer": f"Sample response for: {question}"
    }