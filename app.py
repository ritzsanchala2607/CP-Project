from flask import Flask, render_template, request, send_from_directory, session, redirect, url_for
from PIL import Image
import os, torch, cv2, mediapipe as mp
from transformers import SamModel, SamProcessor, logging as hf_logging
from torchvision import transforms
from diffusers.utils import load_image
from flask_cors import CORS 
import json
import time

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
CORS(app)

# Enable Hugging Face logs
hf_logging.set_verbosity_info()

UPLOAD_FOLDER = '/tmp/uploads'
OUTPUT_FOLDER = '/tmp/outputs'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Global model vars
model, processor, device = None, None, None

def load_model():
    """Load SAM model (CPU only)."""
    global model, processor, device
    device = "cpu"
    print(f"[INFO] Using device: {device}")
    model = SamModel.from_pretrained(
        "Zigeng/SlimSAM-uniform-50",
        cache_dir="/tmp/.cache",
        torch_dtype=torch.float32,
    )
    processor = SamProcessor.from_pretrained("Zigeng/SlimSAM-uniform-50", cache_dir="/tmp/.cache")
    print("[INFO] Model loaded successfully!")

def cleanup_temp_files():
    import shutil
    try:
        if os.path.exists("/tmp/.cache"):
            shutil.rmtree("/tmp/.cache")
        print("[INFO] Cleaned up cache")
    except Exception as e:
        print(f"[WARNING] Cleanup failed: {e}")

def cleanup_old_outputs():
    try:
        if os.path.exists(OUTPUT_FOLDER):
            for file in os.listdir(OUTPUT_FOLDER):
                file_path = os.path.join(OUTPUT_FOLDER, file)
                if os.path.isfile(file_path) and (time.time() - os.path.getctime(file_path) > 3600):
                    os.remove(file_path)
                    print(f"[INFO] Removed old file: {file}")
    except Exception as e:
        print(f"[WARNING] Cleanup outputs failed: {e}")

@app.before_request
def log_request_info():
    print(f"[INFO] Incoming request: {request.method} {request.path}")

@app.route('/health')
def health():
    return "OK", 200

@app.route('/outputs/<filename>')
def serve_output(filename):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(file_path):
        return "File not found", 404
    if filename.lower().endswith(('.jpg', '.jpeg')):
        mimetype = 'image/jpeg'
    elif filename.lower().endswith('.png'):
        mimetype = 'image/png'
    else:
        mimetype = 'application/octet-stream'
    return send_from_directory(OUTPUT_FOLDER, filename, mimetype=mimetype)

@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

def detect_upper_body_coordinates(person_path, is_female=False):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    image = cv2.imread(person_path)
    if image is None:
        raise Exception("No image detected.")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if not results.pose_landmarks:
        raise Exception("No pose detected.")
    h, w, _ = image.shape
    lm = results.pose_landmarks.landmark
    
    # Get key body landmarks
    left_shoulder = (int(lm[11].x * w), int(lm[11].y * h))
    right_shoulder = (int(lm[12].x * w), int(lm[12].y * h))
    left_hip = (int(lm[23].x * w), int(lm[23].y * h))
    right_hip = (int(lm[24].x * w), int(lm[24].y * h))
    
    # Calculate centers
    shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) // 2
    shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) // 2
    hip_center_x = (left_hip[0] + right_hip[0]) // 2
    hip_center_y = (left_hip[1] + right_hip[1]) // 2
    
    if is_female:
        # For girls: Focus on chest-to-hip area to avoid hair on shoulders
        # Calculate chest points (below shoulders to avoid hair)
        chest_offset = 80  # pixels below shoulders
        left_chest_x = left_shoulder[0]
        left_chest_y = left_shoulder[1] + chest_offset
        right_chest_x = right_shoulder[0]
        right_chest_y = right_shoulder[1] + chest_offset
        
        # Chest center point
        chest_center_x = (left_chest_x + right_chest_x) // 2
        chest_center_y = (left_chest_y + right_chest_y) // 2
        
        # Waist center point (between chest and hips)
        waist_center_x = chest_center_x
        waist_center_y = chest_center_y + (hip_center_y - chest_center_y) // 2
        
        coords = {
            "left_chest": (left_chest_x, left_chest_y),
            "right_chest": (right_chest_x, right_chest_y),
            "left_hip": left_hip,
            "right_hip": right_hip,
            "chest_center": (chest_center_x, chest_center_y),
            "waist_center": (waist_center_x, waist_center_y),
            "detection_type": "female_chest_to_hip"
        }
    else:
        # For boys: Use shoulder-based detection (original approach)
        chest_center_x = shoulder_center_x
        chest_center_y = shoulder_center_y + (hip_center_y - shoulder_center_y) // 3
        waist_center_x = shoulder_center_x
        waist_center_y = shoulder_center_y + 2 * (hip_center_y - shoulder_center_y) // 3
        
        coords = {
            "left_shoulder": left_shoulder,
            "right_shoulder": right_shoulder,
            "left_hip": left_hip,
            "right_hip": right_hip,
            "chest_center": (chest_center_x, chest_center_y),
            "waist_center": (waist_center_x, waist_center_y),
            "shoulder_center": (shoulder_center_x, shoulder_center_y),
            "detection_type": "male_shoulder_based"
        }
    
    return coords

@app.route('/', methods=['GET', 'POST'])
def index():
    start_time = time.time()
    if request.method == 'POST':
        try:
            load_model()

            # Person image handling
            use_cached_person = 'person_coordinates' in session and 'person_image_path' in session
            cached_person_flag = use_cached_person
            person_path = None
            person_disk_path = os.path.join(UPLOAD_FOLDER, 'person.jpg')
            
            # Get gender selection from form
            is_female = request.form.get('gender') == 'female'
            print(f"[INFO] Gender selection: {'Female' if is_female else 'Male'}")

            if use_cached_person:
                person_path = session['person_image_path']
                person_coordinates = session['person_coordinates']
                print(f"[INFO] Using cached person {person_path}")
            else:
                person_file = request.files.get('person_image')
                if person_file and person_file.filename != '':
                    person_path = person_disk_path
                    person_file.save(person_path)
                    print(f"[INFO] Saved new person {person_path}")
                elif os.path.exists(person_disk_path):
                    person_path = person_disk_path
                    print(f"[INFO] Reusing person {person_path}")
                else:
                    return "No person image provided."

                # Detect pose with gender-aware coordinates
                coords = detect_upper_body_coordinates(person_path, is_female=is_female)
                person_coordinates = coords.copy()  # Use all coordinates from detection

                # Cache
                session['person_image_path'] = person_path
                session['person_coordinates'] = person_coordinates
                cached_person_flag = True

            # Garment handling
            tshirt_file = request.files['tshirt_image']
            tshirt_path = os.path.join(UPLOAD_FOLDER, 'tshirt.png')
            tshirt_file.save(tshirt_path)

            # Inference
            img = load_image(person_path)
            new_tshirt = load_image(tshirt_path)
            
            # Use cached or fresh coordinates
            coords = person_coordinates
            detection_type = coords.get('detection_type', 'male_shoulder_based')
            
            # Create input points based on detection type
            if detection_type == 'female_chest_to_hip':
                # For girls: Use chest-to-hip points (avoiding shoulders/hair)
                input_points = [[
                    list(coords['left_chest']),
                    list(coords['right_chest']),
                    list(coords['chest_center']),
                    list(coords['waist_center']),
                    list(coords['left_hip']),
                    list(coords['right_hip'])
                ]]
                print("[INFO] Using female chest-to-hip detection (avoiding hair on shoulders)")
            else:
                # For boys: Use shoulder-based points
                input_points = [[
                    list(coords['left_shoulder']),
                    list(coords['right_shoulder']),
                    list(coords['chest_center']),
                    list(coords['waist_center']),
                    list(coords['left_hip']),
                    list(coords['right_hip']),
                    list(coords['shoulder_center'])
                ]]
                print("[INFO] Using male shoulder-based detection")

            inputs = processor(img, input_points=input_points, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            masks = processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu()
            )
            mask_tensor = masks[0][0][2].to(dtype=torch.uint8)
            mask = transforms.ToPILImage()(mask_tensor * 255)

            # Merge
            new_tshirt = new_tshirt.resize(img.size, Image.LANCZOS)
            img_with_new_tshirt = Image.composite(new_tshirt, img, mask)
            result_path = os.path.join(OUTPUT_FOLDER, 'result.jpg')
            os.makedirs(OUTPUT_FOLDER, exist_ok=True)
            img_with_new_tshirt.save(result_path)

            # Unique result name
            import uuid, shutil
            unique_filename = f"result_{uuid.uuid4().hex[:8]}.jpg"
            unique_result_path = os.path.join(OUTPUT_FOLDER, unique_filename)
            shutil.copy2(result_path, unique_result_path)

            processing_time = time.time() - start_time
            cleanup_old_outputs()

            return render_template(
                'index.html',
                result_img=f'/outputs/{unique_filename}',
                cached_person=cached_person_flag,
                person_image_path=person_path,
                processing_time=f"{processing_time:.2f}s"
            )
        except Exception as e:
            print(f"[ERROR] {e}")
            return f"Error: {e}"

    # GET
    has_cached = 'person_coordinates' in session and 'person_image_path' in session
    return render_template(
        'index.html',
        cached_person=has_cached,
        person_image_path=session.get('person_image_path') if has_cached else None
    )

@app.route('/change_person', methods=['POST'])
def change_person():
    session.pop('person_coordinates', None)
    session.pop('person_image_path', None)
    try:
        for f in ['person.jpg', 'tshirt.png']:
            path = os.path.join(UPLOAD_FOLDER, f)
            if os.path.exists(path):
                os.remove(path)
        for f in os.listdir(OUTPUT_FOLDER):
            fp = os.path.join(OUTPUT_FOLDER, f)
            if os.path.isfile(fp):
                os.remove(fp)
        print("[INFO] Cleared person data")
    except Exception as e:
        print(f"[WARNING] Failed to clear: {e}")
    return redirect(url_for('index'))

@app.route('/cleanup', methods=['POST'])
def cleanup():
    cleanup_temp_files()
    cleanup_old_outputs()
    return "Cleanup completed", 200

@app.route('/test-image')
def test_image():
    from PIL import ImageDraw
    img = Image.new('RGB', (200, 200), color='red')
    draw = ImageDraw.Draw(img)
    draw.text((50, 100), "TEST", fill='white')
    test_path = os.path.join(OUTPUT_FOLDER, 'test.jpg')
    img.save(test_path)
    return f'<img src="/outputs/test.jpg" alt="Test">'

if __name__ == '__main__':
    print("[INFO] Starting Flask server...")
    app.run(debug=True, host='0.0.0.0')
