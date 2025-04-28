from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from scraping import get_website_content

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# (Model loading kept for showing purpose, but not really used)
try:
    from llama_cpp import Llama
    model_path = os.path.join(os.path.dirname(__file__), "model", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    llm = Llama(model_path=model_path, n_ctx=2048, n_threads=8)
    print("✅ Llama model loaded successfully!")
except Exception as e:
    print(f"⚠️ Warning: Model not loaded: {e}")
    llm = None

website_data = get_website_content()

class QueryRequest(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"message": "Welcome to Nova Tech Solutions chatbot backend! 🚀"}

@app.post("/chat")
def chat(request: QueryRequest):
    user_message = request.message.lower().strip()
    print(f"Received: {user_message}")

    # Static simple logic for important questions
    if any(keyword in user_message for keyword in ["hello", "hi", "hey"]):
        return {"response": "Hello! 👋 How can I assist you today?"}

    if "services" in user_message:
        return {"response": "We offer Web Development, AI Integration, Mobile App Development, UI/UX Design, Cloud Solutions, and Digital Marketing. Feel free to ask more!"}

    if "web development" in user_message:
        return {"response": "Our web development team builds modern, responsive websites tailored to your business needs."}

    if "ai integration" in user_message:
        return {"response": "We specialize in integrating AI solutions to automate workflows and boost efficiency."}

    if "mobile app development" in user_message:
        return {"response": "We create custom mobile apps for Android and iOS, delivering great user experience and performance."}

    if "ui/ux design" in user_message:
        return {"response": "Our UI/UX experts design intuitive, user-centered interfaces that engage your customers."}

    if "cloud solutions" in user_message:
        return {"response": "We offer scalable and secure cloud computing services to help you grow your business."}

    if "digital marketing" in user_message:
        return {"response": "We help grow your brand through SEO, PPC advertising, social media marketing, and more."}

    if "mission" in user_message or "about" in user_message:
        return {"response": "At Nova Tech Solutions, we deliver innovative and secure IT solutions that empower businesses to succeed."}

    if "contact" in user_message or "support" in user_message:
        return {"response": website_data.get("contact_info", "You can reach us via the contact form on our website.")}

    if "hr" in user_message or "openings" in user_message:
        return {"response": "Thank you for your interest! Please check the Careers page on our website for any current openings."}

    # Default fallback for unknown questions
    return {"response": "Thank you for your query! A representative will get back to you soon with more information. 🚀"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
