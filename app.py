# main.py (at project root)
from fastapi import FastAPI
from api.chat_router import router

app = FastAPI()
app.include_router(router)

@app.get("/")
def welcome():
    """Welcome Function.
    """
    return "Welcome to the project Orphic."


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", "127.0.0.1", 8000)