from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_cpp import Llama
import os
import requests
import asyncio

# Initialize app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model path
model_path = os.path.join(os.path.dirname(__file__), "model", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

# Download model if missing
def download_model():
    if not os.path.exists(model_path):
        print("Model not found. Downloading...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        url = "https://www.dropbox.com/scl/fi/d7gbpkz385t58y5wm8wqu/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf?rlkey=uzqxm8o0u8jxq49op01aamldb&st=rmkwdeli&dl=1"
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(model_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("✅ Model downloaded successfully.")
        else:
            print("❌ Failed to download model.")

download_model()

# Load model
llm = Llama(
    model_path=model_path,
    n_ctx=1024,
    n_threads=8,
)
print("✅ Llama model loaded successfully!")

# Request body
class QueryRequest(BaseModel):
    message: str

@app.get("/")
async def read_root():
    return {"message": "Welcome to NovaTech chatbot!"}

@app.post("/chat")
async def chat(request: QueryRequest):
    user_message = request.message.lower().strip()
    print(f"📥 User sent: {user_message}")

    # Quick static replies
    if any(word in user_message for word in ["hello", "hi", "hey"]):
        return {"response": "Hello! 👋 How can I assist you today?"}
    if "services" in user_message:
        return {"response": "We offer Web Development, AI, Mobile Apps, Cloud Solutions, and Digital Marketing!"}

    # Model-based reply for everything else
    prompt = f"""
You are NovaTech's AI Assistant. Help users about NovaTech Solutions.
Answer only about services, careers, company info.
If question unrelated, say: "Sorry, I can only answer about NovaTech Solutions."

User: {request.message}
Assistant:
"""
    try:
        # Use timeout to prevent server crash
        response = await asyncio.wait_for(run_model(prompt), timeout=30)
        return {"response": response}
    except asyncio.TimeoutError:
        print("⚠️ Model took too long.")
        return {"response": "Sorry, the server is busy. Please try again."}
    except Exception as e:
        print(f"❌ Model error: {e}")
        return {"response": "Sorry, something went wrong. Please try again later."}

async def run_model(prompt):
    output = llm(
        prompt=prompt,
        temperature=0.3,
        max_tokens=150,
        stop=["User:", "Assistant:"]
    )
    response_text = output["choices"][0]["text"].strip()
    print(f"✅ Model replied: {response_text}")
    return response_text
