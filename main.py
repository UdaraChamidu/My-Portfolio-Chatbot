from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.genai as genai
from google.genai import types
import os
import dotenv 

dotenv.load_dotenv()

app = FastAPI()

# Allow frontend (localhost + deployed Vercel) to talk with backend
origins = [
    "http://localhost:5173",   # local dev
    "https://udara-chamidu-portfolio.vercel.app",  #  deployed frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Google Gen AI client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat(req: ChatRequest):
    # Create a chat instance with the Gemini 2.5 Flash model
    chat = client.aio.chats.create(model="gemini-2.5-flash")
    
    # Send the user's message to the model
    response = await chat.send_message(req.message)
    
    # Return the model's response
    return {"response": response.text}
