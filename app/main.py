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
    allow_origins=["*"],  # You can replace * with your domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to model
model_path = os.path.join(os.path.dirname(__file__), "model", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

# Google Drive direct download link
# ❗ Replace YOUR_FILE_ID with your actual Google Drive file ID
model_download_url = "https://drive.google.com/uc?export=download&id=1jRN4EBzmui1RNieP1crTuSCv70aB9d1b"

# Function to download the model if it doesn't exist
def download_model():
    if not os.path.exists(model_path):
        print("Model not found. Downloading from Google Drive...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        response = requests.get(model_download_url, stream=True)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Model downloaded successfully.")
        else:
            print(f"Failed to download model. Status code: {response.status_code}")

# Download model if missing
download_model()

# Load Llama model
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=8,
)

print("Llama model loaded successfully!")

# Static content scraped from website
website_data = get_website_content()

class QueryRequest(BaseModel):
    message: str

# Home route
@app.get("/")
def read_root():
    return {"message": "Welcome to Nova Tech Solutions chatbot backend! 🚀"}

# Chatbot route
@app.post("/chat")
def chat(request: QueryRequest):
    user_message = request.message.lower().strip()
    print(f"Received message: {user_message}")

    # Check for greetings and respond accordingly
    if "hello" in user_message or "hi" in user_message or "hey" in user_message:
        return {"response": "Hello! 👋 How can I assist you today?"}

    # General services response
    if "services" in user_message:
        return {"response": "We offer a variety of services designed to elevate your business: Web Development, AI Integration, Mobile App Development, UI/UX Design, Cloud Solutions, and Digital Marketing. Feel free to ask about any specific service!"}
    
    # Elaborate on specific services if requested
    if "web development" in user_message:
        return {"response": "Our web development team builds responsive, fast, and modern websites to help elevate your brand. We focus on user-friendly design and seamless performance."}
    
    elif "ai integration" in user_message:
        return {"response": "We integrate AI into your business processes to help automate tasks, improve efficiency, and drive innovation."}
    
    elif "mobile app development" in user_message:
        return {"response": "We offer custom mobile app development solutions for both iOS and Android platforms, tailored to meet your specific business needs."}
    
    elif "ui/ux design" in user_message:
        return {"response": "Our UI/UX design team focuses on creating user-centered designs that ensure a seamless and engaging experience for your customers."}
    
    elif "cloud solutions" in user_message:
        return {"response": "We provide scalable cloud services to support your business growth, ensuring your operations are smooth and secure in the cloud."}
    
    elif "digital marketing" in user_message:
        return {"response": "We help promote your brand through effective digital marketing strategies, including SEO, social media, and online campaigns."}

    # Check if the message matches known categories first
    elif "mission" in user_message or "about" in user_message:
        return {"response": "At Nova Tech Solutions, our mission is to deliver innovative and secure IT solutions that empower businesses to succeed and grow in an ever-evolving digital landscape."}
    
    elif "contact" in user_message or "support" in user_message:
        return {"response": website_data["contact_info"]}

    # Refined response for HR/mission-related queries
    elif "hr" in user_message or "mission statement" in user_message:
        return {"response": "At Nova Tech Solutions, our mission is to deliver innovative and secure IT solutions that empower businesses to succeed and grow in an ever-evolving digital landscape."}
    
    else:
        # If no match, use Llama model with improved prompt
        prompt = f"""
You are a professional assistant for Nova Tech Solutions.
Answer in a clear, confident, and professional tone, ensuring the response is polite and informative.
Only answer if the user's question is about Nova Tech Solutions' services, products, support, or careers.
Otherwise, politely say: "I'm sorry, I can only answer questions about Nova Tech Solutions."

Examples:

User: "What services do you offer?"
Assistant: "We offer a range of services including IT consulting, software development, cloud solutions, and technical support."

User: "What is your mission statement?"
Assistant: "At Nova Tech Solutions, our mission is to deliver innovative and secure IT solutions that empower businesses to succeed and grow in an ever-evolving digital landscape."

Now answer:

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
