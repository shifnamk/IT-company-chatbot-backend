from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_cpp import Llama
import os
import requests
from scraping import get_website_content  # Import scraping function

# Initialize FastAPI app
app = FastAPI()

# CORS setup (allow frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to model
model_path = os.path.join(os.path.dirname(__file__), "model", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

# Dropbox download link (make sure ?dl=1 for direct download)
model_download_url = "https://www.dropbox.com/scl/fi/d7gbpkz385t58y5wm8wqu/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf?rlkey=uzqxm8o0u8jxq49op01aamldb&dl=1"

# Function to download model if missing
def download_model():
    if not os.path.exists(model_path):
        print("Model not found. Downloading...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        response = requests.get(model_download_url, stream=True)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Model downloaded successfully.")
        else:
            raise Exception(f"Failed to download model. Status code: {response.status_code}")

# Ensure model is available
download_model()

# Load Llama model
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=8,
)

print("✅ Llama model loaded successfully!")

# Static website data
website_data = get_website_content()

# Request body schema
class QueryRequest(BaseModel):
    message: str

# Home route
@app.get("/")
def read_root():
    return {"message": "Welcome to Nova Tech Solutions chatbot backend! 🚀"}

# Chat route
@app.post("/chat")
def chat(request: QueryRequest):
    user_message = request.message.lower().strip()
    print(f"Received message: {user_message}")

    # Static FAQ-like responses
    if any(word in user_message for word in ["hello", "hi", "hey"]):
        return {"response": "Hello! 👋 How can I assist you today?"}

    if "services" in user_message:
        return {"response": "We offer Web Development, AI Integration, Mobile App Development, UI/UX Design, Cloud Solutions, and Digital Marketing. Ask about any!"}

    if "web development" in user_message:
        return {"response": "We build modern, responsive websites tailored to your business needs."}

    if "ai integration" in user_message:
        return {"response": "We integrate AI technologies to automate, optimize, and innovate your business operations."}

    if "mobile app" in user_message:
        return {"response": "We develop mobile apps for iOS and Android with a focus on performance and user experience."}

    if "ui/ux" in user_message:
        return {"response": "Our UI/UX experts create beautiful and intuitive designs to maximize user satisfaction."}

    if "cloud" in user_message:
        return {"response": "We provide scalable cloud computing solutions to future-proof your business."}

    if "digital marketing" in user_message:
        return {"response": "We boost your online presence with SEO, social media, and strategic digital campaigns."}

    if "mission" in user_message or "about" in user_message:
        return {"response": "Our mission is to deliver secure, innovative IT solutions that empower businesses in the digital era."}

    if "contact" in user_message or "support" in user_message:
        return {"response": website_data.get("contact_info", "You can reach us via our Contact page.")}

    # Otherwise, use Llama model to generate an answer
    prompt = f"""
You are a professional assistant for Nova Tech Solutions.
Answer clearly, confidently, and politely. Only discuss Nova Tech Solutions' services, products, careers, or support.

User: "{request.message}"
Assistant:
"""

    try:
        output = llm(
            prompt=prompt,
            temperature=0.2,
            max_tokens=180,
            stop=["User:", "Assistant:"]
        )
        response_text = output["choices"][0]["text"].strip()
        print(f"Generated response: {response_text}")
        return {"response": response_text}
    
    except Exception as e:
        print(f"Error: {e}")
        return {"response": "Sorry, something went wrong. Please try again later."}

# --- VERY IMPORTANT: Start server correctly ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))  # Use Render's dynamic port, default 10000 for local
    uvicorn.run(app, host="0.0.0.0", port=port)
