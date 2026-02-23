import queue
import sys
import json
import time
import os
import re
import threading
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from groq import Groq, RateLimitError
from dotenv import load_dotenv

log_queue = queue.Queue()
stop_event = threading.Event()

def custom_print(*args, **kwargs):
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    msg = sep.join(str(a) for a in args) + end
    log_queue.put(msg)
    sys.stdout.write(msg)
    sys.stdout.flush()

print = custom_print

# Load environment variables from .env.local
load_dotenv(dotenv_path=".env.local")

# --- CONFIGURATION: GROQ ---
groq_keys_env = os.getenv("GROQ_API_KEYS", "")
GROQ_API_KEYS = [k.strip() for k in groq_keys_env.split(",") if k.strip()]

if not GROQ_API_KEYS:
    print("⚠️  WARNING: No 'GROQ_API_KEYS' found in .env.local.")
    GROQ_API_KEYS = []

# --- CONFIGURATION: YOUTUBE ---
youtube_keys_env = os.getenv("YOUTUBE_API_KEYS", "")
if not youtube_keys_env:
    youtube_keys_env = os.getenv("YOUTUBE_API_KEY", "")

YOUTUBE_API_KEYS = [k.strip() for k in youtube_keys_env.split(",") if k.strip()]

if not YOUTUBE_API_KEYS:
    print("⚠️  WARNING: No 'YOUTUBE_API_KEYS' found.")
    YOUTUBE_API_KEYS = []

INPUT_FILENAME = "input.json"
OUTPUT_FILENAME = "output.json"

# Set this to True to re-generate solutions even if they exist in the file
FORCE_REGENERATE_SOLUTIONS = True

# --- GLOBAL STATE ---

# Groq State
current_groq_key_index = 0
client = None
if GROQ_API_KEYS:
    client = Groq(api_key=GROQ_API_KEYS[current_groq_key_index])
MODEL_NAME = "llama-3.3-70b-versatile"

# YouTube State
current_yt_key_index = 0
youtube = None
if YOUTUBE_API_KEYS:
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEYS[current_yt_key_index])

# --- HELPER FUNCTIONS ---

def switch_groq_api_key():
    """Switches to the next available Groq API key."""
    global client, current_groq_key_index
    
    if not GROQ_API_KEYS:
        return None

    current_groq_key_index = (current_groq_key_index + 1) % len(GROQ_API_KEYS)
    new_key = GROQ_API_KEYS[current_groq_key_index]
    
    print(f"\n      🔄  Switching to Groq API Key #{current_groq_key_index + 1}...")
    log_queue.put(json.dumps({"type": "api_key", "service": "Groq", "current": current_groq_key_index + 1, "total": len(GROQ_API_KEYS), "status": "Switching"}))
    client = Groq(api_key=new_key)
    return new_key

def switch_youtube_key():
    """Switches to the next available YouTube API key."""
    global youtube, current_yt_key_index
    
    if not YOUTUBE_API_KEYS:
        return None

    current_yt_key_index = (current_yt_key_index + 1) % len(YOUTUBE_API_KEYS)
    new_key = YOUTUBE_API_KEYS[current_yt_key_index]
    
    print(f"\n      🔄  Switching to YouTube API Key #{current_yt_key_index + 1}...")
    log_queue.put(json.dumps({"type": "api_key", "service": "YouTube", "current": current_yt_key_index + 1, "total": len(YOUTUBE_API_KEYS), "status": "Switching"}))
    youtube = build('youtube', 'v3', developerKey=new_key)
    return youtube

def generate_with_retry(messages_list, task_name="Generation", max_retries=10, **kwargs):
    """
    Accepts a list of messages (System/User) to prevent hallucinations.
    """
    global client 
    
    if not client:
        print("❌ [Error] No Groq client initialized.")
        return None

    keys_tried_in_this_attempt = 0

    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=messages_list,
                model=MODEL_NAME,
                temperature=0.3, # Low temperature to reduce hallucination
                top_p=0.8,
                **kwargs 
            )
            return chat_completion.choices[0].message.content.strip()
            
        except RateLimitError:
            print(f"\n      ⚠️  [Groq Rate Limit] Key #{current_groq_key_index + 1} exhausted.")
            log_queue.put(json.dumps({"type": "api_key", "service": "Groq", "current": current_groq_key_index + 1, "total": len(GROQ_API_KEYS), "status": "Exhausted"}))
            
            switch_groq_api_key()
            keys_tried_in_this_attempt += 1

            if keys_tried_in_this_attempt >= len(GROQ_API_KEYS):
                print(f"      🛑  All Groq keys exhausted. Cooling down for 20s...")
                time.sleep(20)
                keys_tried_in_this_attempt = 0 
            else:
                time.sleep(1) 
            
            print(f"      🔄  Retrying with new key...")
            continue
            
        except Exception as e:
            print(f"\n      ❌  [Groq Error] {e}")
            time.sleep(5)
            continue
    
    print("\n      ❌  [Failed] Max retries exceeded.")
    return None

def get_youtube_search_query(question_text):
    try:
        search_text = question_text.split('/')[0].strip()
        # Simple prompt for query generation
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates YouTube search queries."},
            {"role": "user", "content": f"Write a specific YouTube search query for: \"{search_text}\". Only return the query string."}
        ]
        return generate_with_retry(messages, task_name="Query Gen")
    except Exception as e:
        print(f"      ❌  [Error] {e}")
        return None

def is_comparison_question(text):
    """
    Detects if a question requires a comparison table.
    """
    text_lower = text.lower()
    keywords = [
        "compare", "difference", "distinguish", "versus", " vs ", 
        "differentiate", "comparison", "similarities", "contrast"
    ]
    return any(keyword in text_lower for keyword in keywords)

def get_detailed_solution(question_text, unit_title, marks_history):
    try:
        # 1. Determine Marks/Depth
        avg_marks = 7
        if marks_history:
            try:
                marks_list = [int(h.get('marks', 7)) for h in marks_history]
                if marks_list:
                    avg_marks = sum(marks_list) / len(marks_list)
            except:
                pass
        
        depth_instruction = "Provide a comprehensive answer (approx. 2-3 pages)."
        if avg_marks < 5:
            depth_instruction = "Provide a concise answer (3-4 marks)."
        elif avg_marks > 10:
            depth_instruction = "Provide an extensive, in-depth answer (13 marks)."

        # 2. Check for Comparison Requirement
        is_comparison = is_comparison_question(question_text)
        
        formatting_override = ""
        if is_comparison:
            formatting_override = (
                "\n\n🚨 **MANDATORY FORMATTING FOR THIS QUESTION** 🚨\n"
                "This question asks for a COMPARISON or DIFFERENCE.\n"
                "1. You MUST present the core differences in a **Markdown Table**.\n"
                "2. The table must have clear columns (e.g., Parameter | Concept A | Concept B).\n"
                "3. Do NOT write the differences as paragraphs. Use the table."
            )

        # 3. Construct System Prompt (The Rules)
        # This separates rules from content to prevent "hallucinating the prompt back"
        system_prompt = (
            "You are an expert Engineering Professor for Sant Gadge Baba Amravati University.\n"
            "Your task is to write detailed, academic exam solutions.\n\n"
            "**CORE RULES (STRICT ADHERENCE REQUIRED):**\n"
            "1. **NO REPETITION:** Do NOT repeat the question. Do NOT output your internal instructions. Start the answer immediately.\n"
            "2. **MATH FORMATTING:** Use `$$` for ALL distinct equations. \n"
            "   - Syntax: `$$ equation $$` (with blank lines before and after).\n"
            "   - NEVER use inline math `$` for formulas.\n"
            "3. **DERIVATIONS:** Show every step clearly on new lines.\n"
            "4. **STRUCTURE:** Use `## Headings` and bullet points.\n"
            "5. **DIAGRAMS:** If needed, use Mermaid.js syntax inside ```mermaid blocks.\n"
            f"{formatting_override}" 
        )

        # 4. Construct User Prompt (The Content)
        user_prompt = (
            f"**Subject Unit:** {unit_title}\n"
            f"**Target Depth:** {depth_instruction}\n\n"
            f"**Question:**\n{question_text}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        return generate_with_retry(messages, task_name="Solution Gen", max_tokens=4096)
    except Exception as e:
        print(f"      ❌  [Error] {e}")
        return None

def search_youtube_video(query):
    global youtube
    
    if not query: return None
    if not youtube: 
        print("      ❌  [Error] No YouTube client initialized.")
        return None

    keys_tried = 0
    max_attempts = len(YOUTUBE_API_KEYS) if YOUTUBE_API_KEYS else 1
    
    while keys_tried < max_attempts:
        try:
            request = youtube.search().list(
                part="snippet", 
                maxResults=1, 
                q=query, 
                type="video", 
                videoEmbeddable="true"
            )
            response = request.execute()
            items = response.get('items', [])
            
            if not items: return None
            
            video = items[0]
            return {
                "videoId": video['id']['videoId'],
                "title": video['snippet']['title'],
                "channelTitle": video['snippet']['channelTitle'],
                "searchQuery": query
            }

        except HttpError as e:
            if e.resp.status in [403, 400]:
                print(f"\n      ⚠️  [YouTube Error {e.resp.status}] Key #{current_yt_key_index + 1} failed.")
                if len(YOUTUBE_API_KEYS) > 1:
                    switch_youtube_key()
                    keys_tried += 1
                else:
                    print("      🛑  No other keys available.")
                    return None
            else:
                print(f"      ❌  [YouTube API Error] {e}")
                return None
        except Exception as e:
             print(f"      ❌  [YouTube Unknown Error] {e}")
             return None
    
    print("      🛑  All YouTube API keys exhausted for this query.")
    return None

def save_json_file(filename, data):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f" [Error Saving File] {e}")

def count_total_questions(data):
    return sum(len(unit.get('questions', [])) for unit in data.get('units', []))

def process_subject(data):
    subject_title = data.get('title', 'Unknown Subject')
    total_qs = count_total_questions(data)
    global_counter = 0
    anomalies_list = []

    print("="*60)
    print(f" 🚀 STARTING PROCESSING: {subject_title}")
    print(f" 📊 Total Questions Found: {total_qs}")
    print("="*60)
    
    # Send structured connection start event
    log_queue.put(json.dumps({"type": "progress", "current_q": 0, "total_qs": total_qs}))
    
    # Send initial API Key states
    log_queue.put(json.dumps({"type": "api_key", "service": "Groq", "current": current_groq_key_index + 1, "total": len(GROQ_API_KEYS), "status": "Active"}))
    log_queue.put(json.dumps({"type": "api_key", "service": "YouTube", "current": current_yt_key_index + 1, "total": len(YOUTUBE_API_KEYS), "status": "Active"}))
    
    for u_idx, unit in enumerate(data.get('units', [])):
        if stop_event.is_set():
            print("🛑 Process Cancelled by User.")
            return data
            
        unit_title = unit.get('title', f"Unit {u_idx+1}")
        
        for q_idx, question in enumerate(unit.get('questions', [])):
            global_counter += 1
            q_text = question.get('text', 'No Text')
            short_text = (q_text[:60] + '...') if len(q_text) > 60 else q_text
            
            print(f"\n[{global_counter}/{total_qs}] Unit {u_idx+1} | Q{q_idx+1}: \"{short_text}\"")
            
            # Send full question text immediately to UI
            log_queue.put(json.dumps({"type": "q_details", "field": "text", "value": q_text}))
            
            # Update Progress UI
            log_queue.put(json.dumps({
                "type": "progress", 
                "current_q": global_counter, 
                "total_qs": total_qs,
                "active_step": f"Processing Q{global_counter}"
            }))
            
            modified_this_step = False
            start_time_q = time.time()

            # --- STEP 1: VIDEO ---
            if stop_event.is_set(): return data
            
            if not question.get('video'):
                print("   📺 Step 1: Video Search")
                t0 = time.time()
                
                print("       - Generating Search Query...", end="", flush=True)
                log_queue.put(json.dumps({"type": "active_step", "step": f"Generating Query (Q{global_counter})"}))
                search_query = get_youtube_search_query(q_text)
                
                if stop_event.is_set(): return data
                
                if search_query:
                    print(f" Done. ('{search_query}')")
                    log_queue.put(json.dumps({"type": "q_details", "field": "searchQuery", "value": search_query}))
                    print("       - Searching YouTube...", end="", flush=True)
                    log_queue.put(json.dumps({"type": "active_step", "step": f"Searching YouTube (Q{global_counter})"}))
                    video_data = search_youtube_video(search_query)
                    
                    if video_data:
                        question['video'] = video_data
                        print(f" Found! (ID: {video_data['videoId']})")
                        save_json_file(OUTPUT_FILENAME, data)
                        modified_this_step = True
                    else:
                        question['video'] = None
                        print(" No results found.")
                else:
                    print(" Failed to generate query.")
                
                print(f"      ⏱️  Step time: {time.time() - t0:.1f}s")
                log_queue.put(json.dumps({"type": "q_details", "field": "videoTime", "value": f"{time.time() - t0:.1f}s"}))
                time.sleep(1) 
            else:
                print("   ✅ Video: Already exists. Skipping.")

            # --- STEP 2: SOLUTION ---
            if stop_event.is_set(): return data
            
            if FORCE_REGENERATE_SOLUTIONS or not question.get('solution'):
                print("   📝 Step 2: Detailed Solution (Generating...)")
                log_queue.put(json.dumps({"type": "active_step", "step": f"Generating Solution (Q{global_counter}) - Please Wait..."}))
                t0 = time.time()
                
                # Check if it's a comparison to print a status log
                if is_comparison_question(q_text):
                    print("       - Detected Comparison: Enforcing Table Format...")
                else:
                    print("       - Generating with Llama 3.3 70B...", end="", flush=True)
                
                marks_history = question.get('history', [])
                solution_text = get_detailed_solution(q_text, unit_title, marks_history)
                
                if stop_event.is_set(): return data
                
                if solution_text:
                    char_count = len(solution_text)
                    question['solution'] = solution_text
                    
                    anomaly_msg = ""
                    if char_count > 8000:
                        anomaly_msg = f" ⚠️ ANOMALY: Unusually large response ({char_count} chars). Possible hallucination!"
                        anomalies_list.append(f"Unit {u_idx+1} | Q{q_idx+1}: {char_count} chars")
                    
                    print(f" Generated! ({char_count} chars){anomaly_msg}")
                    
                    # Send to UI via q_details
                    log_queue.put(json.dumps({"type": "q_details", "field": "charCount", "value": f"{char_count} chars"}))
                    if anomaly_msg:
                        log_queue.put(json.dumps({"type": "q_details", "field": "anomaly", "value": f"⚠️ Large response ({char_count} chars)"}))
                        
                    save_json_file(OUTPUT_FILENAME, data)
                    modified_this_step = True
                else:
                    print(" Failed to generate solution.")
                
                print(f"      ⏱️  Step time: {time.time() - t0:.1f}s")
                log_queue.put(json.dumps({"type": "q_details", "field": "solutionTime", "value": f"{time.time() - t0:.1f}s"}))
                time.sleep(1) 
            else:
                print("   ✅ Solution: Already exists. Skipping.")

            # Summary for this question
            total_q_time = time.time() - start_time_q
            if modified_this_step:
                print(f"   💾 Progress Saved. Total Q-Time: {total_q_time:.1f}s")
                log_queue.put(json.dumps({"type": "q_details", "field": "totalTime", "value": f"{total_q_time:.1f}s"}))

    print("\n" + "="*60)
    
    if anomalies_list:
        print(" ⚠️ HALLUCINATION WATCHLIST SUMMARY:")
        for anomaly in anomalies_list:
            print(f"    - {anomaly}")
        print("="*60)

    if stop_event.is_set():
        print(" 🛑 PROCESSING STOPPED BY USER")
        log_queue.put(json.dumps({"type": "status", "value": "Stopped"}))
    else:
        print(" 🎉 PROCESSING COMPLETE!")
        log_queue.put(json.dumps({"type": "status", "value": "Complete"}))
    print("="*60)
    return data

def load_json_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file '{filename}' is not valid JSON.")
        return None

def start_processing():
    stop_event.clear()
    data_to_process = None
    
    if os.path.exists(OUTPUT_FILENAME):
        print(f"📂 Resuming from: {OUTPUT_FILENAME}")
        data_to_process = load_json_file(OUTPUT_FILENAME)
    
    if not data_to_process:
        print(f"📂 Loading fresh input: {INPUT_FILENAME}")
        data_to_process = load_json_file(INPUT_FILENAME)

    if data_to_process:
        process_subject(data_to_process)
        save_json_file(OUTPUT_FILENAME, data_to_process)
    else:
        print("❌ Exiting: Could not load data.")
        
    log_queue.put("[DONE]")

if __name__ == "__main__":
    start_processing()
