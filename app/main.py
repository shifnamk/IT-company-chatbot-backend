from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_cpp import Llama
import os
import requests
from scraping import get_website_content  # Import your scraping function

# Initialize FastAPI app
app = FastAPI()

# CORS setup (allow frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
model_path = os.path.join(os.path.dirname(__file__), "model", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

# Model download URL (Dropbox direct link)
model_download_url = "https://www.dropbox.com/scl/fi/d7gbpkz385t58y5wm8wqu/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf?rlkey=uzqxm8o0u8jxq49op01aamldb&st=rmkwdeli&dl=1"

# Download model if not found
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
            print(f"❌ Failed to download model. Status code: {response.status_code}")

# Call download function
download_model()

# Load model
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=8,
)

print("✅ Llama model loaded successfully!")

# Scrape website content
website_data = get_website_content()

# Define request model
class QueryRequest(BaseModel):
    message: str

# Home route
@app.get("/")
def home():
    return {"message": "NovaTech Chatbot API running!"}

# Chat endpoint
@app.post("/chat")
def chat(request: QueryRequest):
    user_message = request.message.lower().strip()
    print(f"📥 Received: {user_message}")

    # --- Hardcoded shortcut responses ---
    if "hello" in user_message or "hi" in user_message or "hey" in user_message:
        return {"response": "Hello! 👋 How can I assist you today?"}
    
    if "services" in user_message:
        return {"response": "We offer services like Web Development, AI Integration, Mobile App Development, UI/UX Design, Cloud Solutions, and Digital Marketing. Feel free to ask about any specific service!"}

    if "contact" in user_message or "support" in user_message:
        return {"response": website_data.get("contact_info", "Please visit our Contact page for support.")}

    # --- Now let the Llama model answer ---
    prompt = f"""
You are an intelligent, professional chatbot working for NovaTech Solutions.

✅ You must answer ONLY questions related to:
- Company services
- Mission and Vision
- Job openings or Careers
- Contact Information

❌ If the question is unrelated, reply politely: "I'm sorry, I can only assist with questions about NovaTech Solutions."

Here are examples:
User: What services do you offer?
Assistant: We offer Web Development, AI Integration, Mobile App Development, UI/UX Design, Cloud Solutions, and Digital Marketing.

User: Are there any current openings?
Assistant: We are always looking for talented individuals. Please visit our Careers page or contact HR for current opportunities.

User: {request.message}
Assistant:
"""

    try:
        output = llm(
            prompt=prompt,
            temperature=0.3,
            max_tokens=250,   # Allow model enough space to generate
            stop=["User:", "Assistant:"]
        )
        response_text = output["choices"][0]["text"].strip()
        print(f"📤 Model Response: {response_text}")
        return {"response": response_text}

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"response": "Sorry, something went wrong. Please try again later."}
