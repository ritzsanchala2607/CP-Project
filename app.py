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

def load_model():
    """Load model on demand (CPU-only to avoid meta tensor/device issues on Spaces)."""
    global model, processor, device
    
    # Force CPU on Spaces to avoid meta tensor errors when moving devices
    device = "cpu"
    print(f"[INFO] Using device: {device}")
    
    print("[INFO] Loading SAM model and processor...")
    model = SamModel.from_pretrained(
        "Zigeng/SlimSAM-uniform-50",
        cache_dir="/tmp/.cache",
        torch_dtype=torch.float32,
    )
    processor = SamProcessor.from_pretrained("Zigeng/SlimSAM-uniform-50", cache_dir="/tmp/.cache")
    
    # Do NOT move model with .to(); keep it on CPU to prevent meta tensor errors
    print("[INFO] Model and processor loaded successfully on CPU!")

def cleanup_temp_files():
    """Clean up temporary files to save storage"""
    try:
        import shutil
        if os.path.exists("/tmp/.cache"):
            shutil.rmtree("/tmp/.cache")
        print("[INFO] Cleaned up temporary cache files")
    except Exception as e:
        print(f"[WARNING] Could not clean up temp files: {e}")

def cleanup_old_outputs():
    """Clean up old output files to save storage"""
    try:
        if os.path.exists(OUTPUT_FOLDER):
            for file in os.listdir(OUTPUT_FOLDER):
                file_path = os.path.join(OUTPUT_FOLDER, file)
                if os.path.isfile(file_path):
                    # Remove files older than 1 hour
                    if time.time() - os.path.getctime(file_path) > 3600:
                        os.remove(file_path)
                        print(f"[INFO] Removed old output file: {file}")
    except Exception as e:
        print(f"[WARNING] Could not clean up old outputs: {e}")

@app.before_request
def log_request_info():
    print(f"[INFO] Incoming request: {request.method} {request.path}")

@app.route('/health')
def health():
    return "OK", 200

# Route to serve outputs dynamically
@app.route('/outputs/<filename>')
def serve_output(filename):
    print(f"[DEBUG] Serving file: {filename} from {OUTPUT_FOLDER}")
    if not os.path.exists(OUTPUT_FOLDER):
        print(f"[ERROR] Output folder does not exist: {OUTPUT_FOLDER}")
        return "Output folder not found", 404
    
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(file_path):
        print(f"[ERROR] File does not exist: {file_path}")
        return "File not found", 404
    
    print(f"[DEBUG] File exists, serving: {file_path}")
    
    # Set proper MIME type for images
    from flask import Response
    if filename.lower().endswith(('.jpg', '.jpeg')):
        mimetype = 'image/jpeg'
    elif filename.lower().endswith('.png'):
        mimetype = 'image/png'
    else:
        mimetype = 'application/octet-stream'
    
    return send_from_directory(OUTPUT_FOLDER, filename, mimetype=mimetype)

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
            cached_person_flag = use_cached_person
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
                cached_person_flag = True

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
            
            # Ensure output directory exists
            os.makedirs(OUTPUT_FOLDER, exist_ok=True)
            
            # Save the result image
            img_with_new_tshirt.save(result_path)
            print(f"[INFO] Result saved to {result_path}")
            
            # Verify file was saved
            if os.path.exists(result_path):
                file_size = os.path.getsize(result_path)
                print(f"[DEBUG] File saved successfully, size: {file_size} bytes")
            else:
                print(f"[ERROR] File was not saved to {result_path}")

            # Calculate processing time
            processing_time = time.time() - start_time
            print(f"[PERF] Total processing time: {processing_time:.2f}s")
            
            # Clean up old files to save storage
            cleanup_old_outputs()
            
            # Generate a unique filename to avoid caching issues
            import uuid
            unique_filename = f"result_{uuid.uuid4().hex[:8]}.jpg"
            unique_result_path = os.path.join(OUTPUT_FOLDER, unique_filename)
            
            # Copy the result to a unique filename
            import shutil
            shutil.copy2(result_path, unique_result_path)
            
            # Serve via dynamic route with cached person info
            return render_template('index.html', 
                                 result_img=f'/outputs/{unique_filename}',
                                 cached_person=cached_person_flag,
                                 person_image_path=person_path,
                                 processing_time=f"{processing_time:.2f}s")

        except Exception as e:
            print(f"[ERROR] {e}")
            return f"Error: {e}"

    # GET request: keep person image visible if available in session
    has_cached = 'person_coordinates' in session and 'person_image_path' in session
    return render_template(
        'index.html',
        cached_person=has_cached,
        person_image_path=session.get('person_image_path') if has_cached else None
    )

@app.route('/change_person', methods=['POST'])
def change_person():
    """Clear cached person data to allow new person upload"""
    session.pop('person_coordinates', None)
    session.pop('person_image_path', None)
    print("[INFO] Cleared cached person data")
    return render_template('index.html')

@app.route('/cleanup', methods=['POST'])
def cleanup():
    """Manual cleanup of temporary files"""
    cleanup_temp_files()
    cleanup_old_outputs()
    return "Cleanup completed", 200

@app.route('/test-image')
def test_image():
    """Test route to check if image serving works"""
    # Create a simple test image
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (200, 200), color='red')
    draw = ImageDraw.Draw(img)
    draw.text((50, 100), "TEST IMAGE", fill='white')
    
    test_path = os.path.join(OUTPUT_FOLDER, 'test.jpg')
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    img.save(test_path)
    
    return f'<img src="/outputs/test.jpg" alt="Test Image">'

if __name__ == '__main__':
    print("[INFO] Starting Flask server...")
    print("[INFO] Model will be loaded on first request to save memory...")
    app.run(debug=True, host='0.0.0.0')
