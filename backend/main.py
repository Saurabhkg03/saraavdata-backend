from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
import threading
import json
import os
import autotagger

app = FastAPI()

# Split Hosting requirement: CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Can be restricted based on NEXT_PUBLIC_API_URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

process_thread = None

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    # Save the uploaded file as input.json
    with open("input.json", "wb") as f:
        f.write(await file.read())
    
    # If a previous output existed, remove it so we start fresh
    if os.path.exists("output.json"):
        os.remove("output.json")
        
    return {"message": "File uploaded successfully", "filename": file.filename}

@app.get("/api/download")
async def download_file():
    if not os.path.exists("output.json"):
        return {"error": "Output file not found. Process may not be complete."}
    return FileResponse(path="output.json", filename="processed_output.json", media_type="application/json")

@app.get("/api/process")
async def process_endpoint():
    global process_thread
    
    # Start thread if not already running
    if not process_thread or not process_thread.is_alive():
        # Clear the queue from any previous runs
        while not autotagger.log_queue.empty():
            autotagger.log_queue.get()
            
        process_thread = threading.Thread(target=autotagger.start_processing, daemon=True)
        process_thread.start()

    def event_stream():
        # Send initial connected message
        yield f"data: {json.dumps({'message': 'Connected to server stream...'})}\n\n"
        while True:
            msg = autotagger.log_queue.get()
            if msg == "[DONE]":
                yield f"data: {json.dumps({'message': '[DONE]'})}\n\n"
                break
            # We yield as JSON string so the frontend can safely parse lines and special characters
            
            # Check if msg is already a JSON string (our structured events)
            try:
                # If it parses as a dict, it's one of our structured progress events
                parsed_msg = json.loads(msg)
                if isinstance(parsed_msg, dict) and "type" in parsed_msg:
                    yield f"data: {msg}\n\n"
                    continue
            except json.JSONDecodeError:
                pass
                
            # Otherwise, wrap it as a standard text message for the terminal
            yield f"data: {json.dumps({'message': msg})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/api/stop")
async def stop_endpoint():
    autotagger.stop_event.set()
    
    # User requested to literally kill the backend to halt long-running LLM requests instantly.
    def kill_backend():
        import time, os, signal
        time.sleep(0.5) # give it a moment to return the 200 OK response
        os.kill(os.getpid(), signal.SIGINT)
        
    threading.Thread(target=kill_backend, daemon=True).start()
    return {"message": "Backend server is halting completely."}
