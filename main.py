from fastapi import FastAPI
from pydantic import BaseModel
import google.genai as genai
from google.genai import types
import os
import dotenv 

dotenv.load_dotenv()

# Initialize the Google Gen AI client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

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
