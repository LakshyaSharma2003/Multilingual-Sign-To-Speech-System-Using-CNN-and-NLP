from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import base64
import json
from googletrans import Translator
from backend.predictor import SignPredictor

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Predictor and Translator
predictor = SignPredictor()
translator = Translator()

class TranslationRequest(BaseModel):
    text: str
    target_lang: str

@app.post("/translate")
async def translate_text(request: TranslationRequest):
    try:
        translated = translator.translate(request.text, dest=request.target_lang)
        return {"original": request.text, "translated": translated.text}
    except Exception as e:
        return {"error": str(e)}

@app.websocket("/ws/predict")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Extract image and mode
            image_data = message.get("image")
            mode = message.get("mode", "ASL")
            
            if not image_data:
                continue
                
            # Decode base64 image
            encoded_data = image_data.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
                
            # Predict
            letter, confidence = predictor.predict(frame, mode=mode)
            
            # Send back prediction
            await websocket.send_json({
                "letter": letter,
                "confidence": confidence
            })
            
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
