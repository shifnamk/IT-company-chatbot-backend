from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_cpp import Llama
import os
import requests
import asyncio
from scraping import get_website_content

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = os.path.join(os.path.dirname(__file__), "model", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
model_download_url = "https://www.dropbox.com/scl/fi/d7gbpkz385t58y5wm8wqu/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf?rlkey=uzqxm8o0u8jxq49op01aamldb&st=rmkwdeli&dl=1"

def download_model():
    if not os.path.exists(model_path):
        print("Model not found. Downloading...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        response = requests.get(model_download_url, stream=True)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("✅ Model downloaded successfully.")
        else:
            print(f"Failed to download model. Status code: {response.status_code}")

# Ensure model is present
download_model()

# Load model (optimized settings)
llm = Llama(
    model_path=model_path,
    n_ctx=1024,     # ✅ Smaller context (save memory)
    n_threads=4,    # ✅ Fewer threads (avoid CPU overload)
    n_batch=128,    # ✅ Optional: reduce batch size
)

print("✅ Llama model loaded successfully!")

website_data = get_website_content()

class QueryRequest(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"message": "Welcome to Nova Tech Solutions chatbot backend!"}

@app.post("/chat")
async def chat(request: QueryRequest):
    user_message = request.message.lower().strip()
    print(f"📥 Received: {user_message}")

    # Static responses
    if any(word in user_message for word in ["hello", "hi", "hey"]):
        return {"response": "Hello! 👋 How can I assist you today?"}

    if "services" in user_message:
        return {"response": "We offer a variety of services designed to elevate your business: Web Development, AI Integration, Mobile App Development, UI/UX Design, Cloud Solutions, and Digital Marketing."}

    if "web development" in user_message:
        return {"response": "Our web development team builds modern, responsive websites with a focus on performance and user experience."}

    if "ai integration" in user_message:
        return {"response": "We specialize in integrating AI solutions to automate and optimize business operations."}

    if "mission" in user_message or "about" in user_message:
        return {"response": "At Nova Tech Solutions, our mission is to deliver innovative IT solutions that empower businesses to grow."}

    if "contact" in user_message or "support" in user_message:
        return {"response": website_data.get("contact_info", "You can contact us via email or our website contact form.")}

    # 🔥 If no static reply, use model
    prompt = f"""
You are a professional assistant for Nova Tech Solutions.
Answer clearly, politely, and only about Nova Tech services or jobs.

User: {request.message}
Assistant:
"""

    try:
        # Add timeout to avoid infinite model hangs
        response = await asyncio.wait_for(run_model(prompt), timeout=25.0)
        return {"response": response}

    except asyncio.TimeoutError:
        print("⚠️ Model response timed out!")
        return {"response": "Sorry, my brain took too long to think! Please try again."}

    except Exception as e:
        print(f"❌ Model Error: {e}")
        return {"response": "Sorry, something went wrong internally. Please try again later."}

# Separate function to call model
async def run_model(prompt):
    output = llm(
        prompt=prompt,
        temperature=0.2,
        max_tokens=180,
        stop=["User:", "Assistant:"]
    )
    response_text = output["choices"][0]["text"].strip()
    print(f"✅ Model replied: {response_text}")
    return response_text
