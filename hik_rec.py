import cv2
import numpy as np
import insightface
import os
import time
import warnings
from hikvisionapi import Client

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Configuration ---

# Model and Database Settings
MODEL_NAME = "buffalo_l"
DB_DIR = "live_face_db"
THRESHOLD = 0.6

# Hikvision Camera Settings
CAMERA_IP = "192.168.29.195"
USERNAME = "admin"
PASSWORD = "ronin@117"
# Using channel 101 for the high-resolution main stream
CHANNEL = 101 

# Display Settings
# The width to which the preview window will be resized.
# The height will be calculated automatically to maintain the aspect ratio.
DISPLAY_WIDTH = 1280

# --- Core Functions (Unchanged) ---

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)

def load_db():
    db = []
    if not os.path.exists(DB_DIR):
        print(f"ERROR: Database directory '{DB_DIR}' not found.")
        return []
    
    print(f"Loading faces from database '{DB_DIR}'...")
    for name in os.listdir(DB_DIR):
        person_dir = os.path.join(DB_DIR, name)
        if not os.path.isdir(person_dir): continue
        for file in os.listdir(person_dir):
            if file.endswith('.npy'):
                try:
                    emb = np.load(os.path.join(person_dir, file))
                    db.append((emb, name))
                except Exception as e:
                    print(f"Could not load file {file}: {e}")
    return db

def find_match(embedding, db):
    if not db: return None, 0.0
    sims = [cosine_sim(embedding, e) for e, _ in db]
    max_idx = np.argmax(sims)
    max_sim = sims[max_idx]
    if max_sim > THRESHOLD:
        return db[max_idx][1], max_sim
    else:
        return "Unknown", max_sim

# --- Main Recognition Logic ---

def recognize_from_hikvision():
    # Initialize the InsightFace analysis tool
    print("Initializing Face Analysis model...")
    app = insightface.app.FaceAnalysis(name=MODEL_NAME, providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Load the known faces
    db = load_db()
    if not db:
        print("Database is empty. Please register faces first.")
        return
    print(f"Database loaded successfully with {len(db)} face embeddings.")

    # Initialize the Hikvision camera client
    print(f"Connecting to camera at {CAMERA_IP}...")
    try:
        cam = Client(f'http://{CAMERA_IP}', USERNAME, PASSWORD, timeout=30)
    except Exception as e:
        print(f"ERROR: Could not connect to camera. {e}")
        return

    # Create a resizable window for the preview
    cv2.namedWindow("Hikvision Face Recognition", cv2.WINDOW_NORMAL)
    print("\nStarting real-time recognition. Press 'q' in the preview window to exit.")

    while True:
        try:
            # 1. Get the FULL RESOLUTION snapshot from the main stream
            response = cam.Streaming.channels[CHANNEL].picture(method='get', type='opaque_data')
            image_bytes = np.asarray(bytearray(response.content), dtype=np.uint8)
            frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

            if frame is None:
                print("Could not decode frame from camera, skipping...")
                continue
            
            # 2. Perform recognition on the FULL RESOLUTION frame for best accuracy
            faces = app.get(frame)
            
            # 3. Draw bounding boxes and labels on the FULL RESOLUTION frame
            for face in faces:
                bbox = face.bbox.astype(int)
                emb = face.embedding
                label, similarity = find_match(emb, db)
                display_text = f"{label} ({similarity:.2f})"
                
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, display_text, (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 4. Resize the processed frame for display
            # Calculate the new height to maintain aspect ratio
            h, w = frame.shape[:2]
            aspect_ratio = h / w
            new_height = int(DISPLAY_WIDTH * aspect_ratio)
            display_frame = cv2.resize(frame, (DISPLAY_WIDTH, new_height), interpolation=cv2.INTER_AREA)

            # 5. Show the RESIZED frame in the window
            cv2.imshow("Hikvision Face Recognition", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"An error occurred in the main loop: {e}")
            print("Attempting to reconnect in 5 seconds...")
            time.sleep(5)

    cv2.destroyAllWindows()
    print("Recognition stopped.")

if __name__ == "__main__":
    recognize_from_hikvision()