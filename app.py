from flask import Flask, render_template, request, send_from_directory, session
from PIL import Image
import os, torch, cv2, mediapipe as mp
from transformers import SamModel, SamProcessor, logging as hf_logging
from torchvision import transforms
from diffusers.utils import load_image
from flask_cors import CORS 
import json
import time

app= Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')  # Change this to a random secret key
CORS(app)

# Enable Hugging Face detailed logs (shows model download progress)
hf_logging.set_verbosity_info()


UPLOAD_FOLDER = '/tmp/uploads'
OUTPUT_FOLDER = '/tmp/outputs'

if not os.path.exists(UPLOAD_FOLDER):
    print(f"[WARN] {UPLOAD_FOLDER} does not exist. Creating...")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

if not os.path.exists(OUTPUT_FOLDER):
    print(f"[WARN] {OUTPUT_FOLDER} does not exist. Creating...")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# Global model variables
model, processor = None, None
device = None

def initialize_model():
    """Initialize model once at startup"""
    global model, processor, device
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    
    print("[INFO] Loading SAM model and processor...")
    model = SamModel.from_pretrained("Zigeng/SlimSAM-uniform-50", cache_dir="/app/.cache")
    processor = SamProcessor.from_pretrained("Zigeng/SlimSAM-uniform-50", cache_dir="/app/.cache")
    
    # Move model to device
    model = model.to(device)
    print(f"[INFO] Model and processor loaded successfully on {device}!")

def load_model():
    """Ensure model is loaded (should already be loaded at startup)"""
    global model, processor
    if model is None or processor is None:
        print("[WARNING] Model not loaded, initializing now...")
        initialize_model()

def warmup_model():
    """Warm up the model with a dummy inference"""
    global model, processor, device
    if model is None or processor is None:
        return
    
    print("[INFO] Warming up model...")
    try:
        # Create a dummy image and points for warmup
        dummy_img = Image.new('RGB', (512, 512), color='white')
        dummy_points = [[[256, 256], [300, 300]]]
        inputs = processor(dummy_img, input_points=dummy_points, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            _ = model(**inputs)
        print("[INFO] Model warmup completed!")
    except Exception as e:
        print(f"[WARNING] Model warmup failed: {e}")

@app.before_request
def log_request_info():
    print(f"[INFO] Incoming request: {request.method} {request.path}")

@app.route('/health')
def health():
    return "OK", 200

# Route to serve outputs dynamically
@app.route('/outputs/<filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

# Route to serve cached person images
@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

def detect_pose_and_get_coordinates(person_path):
    """Extract pose coordinates from person image"""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    image = cv2.imread(person_path)
    if image is None:
        raise Exception("No image detected.")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if not results.pose_landmarks:
        raise Exception("No pose detected.")
    
    height, width, _ = image.shape
    landmarks = results.pose_landmarks.landmark
    left_shoulder = (int(landmarks[11].x * width), int(landmarks[11].y * height))
    right_shoulder = (int(landmarks[12].x * width), int(landmarks[12].y * height))
    
    return left_shoulder, right_shoulder

@app.route('/', methods=['GET', 'POST'])
def index():
    start_time = time.time()
    print(f"[INFO] Handling {request.method} on /")
    if request.method == 'POST':
        try:
            load_model()
            
            # Check if we have a cached person image and coordinates
            use_cached_person = 'person_coordinates' in session and 'person_image_path' in session
            person_coordinates = None
            person_path = None
            
            if use_cached_person:
                # Use cached person image and coordinates
                person_path = session['person_image_path']
                person_coordinates = session['person_coordinates']
                print(f"[INFO] Using cached person image: {person_path}")
                print(f"[INFO] Using cached coordinates: {person_coordinates}")
            else:
                # Process new person image
                person_file = request.files.get('person_image')
                if not person_file or person_file.filename == '':
                    return "No person image provided. Please upload a person image first."
                
                person_path = os.path.join(UPLOAD_FOLDER, 'person.jpg')
                person_file.save(person_path)
                print(f"[INFO] Saved new person image to {person_path}")
                
                # Detect pose and get coordinates
                left_shoulder, right_shoulder = detect_pose_and_get_coordinates(person_path)
                person_coordinates = {
                    'left_shoulder': left_shoulder,
                    'right_shoulder': right_shoulder
                }
                
                # Cache the person image and coordinates
                session['person_image_path'] = person_path
                session['person_coordinates'] = person_coordinates
                print(f"[INFO] Cached person coordinates: {person_coordinates}")

            # Process garment image
            tshirt_file = request.files['tshirt_image']
            tshirt_path = os.path.join(UPLOAD_FOLDER, 'tshirt.png')
            tshirt_file.save(tshirt_path)
            print(f"[INFO] Saved garment image to {tshirt_path}")

            # SAM model inference using cached or new coordinates
            img = load_image(person_path)
            new_tshirt = load_image(tshirt_path)
            input_points = [[[person_coordinates['left_shoulder'][0], person_coordinates['left_shoulder'][1]], 
                           [person_coordinates['right_shoulder'][0], person_coordinates['right_shoulder'][1]]]]
            inputs = processor(img, input_points=input_points, return_tensors="pt")
            
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():  # Disable gradient computation for inference
                outputs = model(**inputs)
            
            masks = processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu()
            )
            mask_tensor = masks[0][0][2].to(dtype=torch.uint8)
            mask = transforms.ToPILImage()(mask_tensor * 255)

            # Combine images
            new_tshirt = new_tshirt.resize(img.size, Image.LANCZOS)
            img_with_new_tshirt = Image.composite(new_tshirt, img, mask)
            result_path = os.path.join(OUTPUT_FOLDER, 'result.jpg')
            img_with_new_tshirt.save(result_path)
            print(f"[INFO] Result saved to {result_path}")

            # Calculate processing time
            processing_time = time.time() - start_time
            print(f"[PERF] Total processing time: {processing_time:.2f}s")
            
            # Serve via dynamic route with cached person info
            return render_template('index.html', 
                                 result_img='/outputs/result.jpg',
                                 cached_person=use_cached_person,
                                 person_image_path=person_path,
                                 processing_time=f"{processing_time:.2f}s")

        except Exception as e:
            print(f"[ERROR] {e}")
            return f"Error: {e}"

    return render_template('index.html')

@app.route('/change_person', methods=['POST'])
def change_person():
    """Clear cached person data to allow new person upload"""
    session.pop('person_coordinates', None)
    session.pop('person_image_path', None)
    print("[INFO] Cleared cached person data")
    return render_template('index.html')

if __name__ == '__main__':
    # Initialize model at startup
    print("[INFO] Initializing model...")
    initialize_model()
    
    # Warm up the model
    warmup_model()
    
    print("[INFO] Starting Flask server...")
    app.run(debug=True, host='0.0.0.0')
