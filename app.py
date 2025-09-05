from flask import Flask, render_template, request, send_from_directory
from PIL import Image
import os, torch, cv2, mediapipe as mp
from transformers import SamModel, SamProcessor, logging as hf_logging
from torchvision import transforms
from diffusers.utils import load_image
from flask_cors import CORS 

app= Flask(__name__)
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


# Lazy-load model
model, processor = None, None

def load_model():
    global model, processor
    if model is None or processor is None:
        print("[INFO] Loading SAM model and processor...")
        model = SamModel.from_pretrained("Zigeng/SlimSAM-uniform-50", cache_dir="/app/.cache")
        processor = SamProcessor.from_pretrained("Zigeng/SlimSAM-uniform-50", cache_dir="/app/.cache")
        print("[INFO] Model and processor loaded successfully!")

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

@app.route('/', methods=['GET', 'POST'])
def index():
    print(f"[INFO] Handling {request.method} on /")
    if request.method == 'POST':
        try:
            load_model()

            # Save uploaded images
            person_file = request.files['person_image']
            tshirt_file = request.files['tshirt_image']
            person_path = os.path.join(UPLOAD_FOLDER, 'person.jpg')
            tshirt_path = os.path.join(UPLOAD_FOLDER, 'tshirt.png')
            person_file.save(person_path)
            tshirt_file.save(tshirt_path)
            print(f"[INFO] Saved files to {UPLOAD_FOLDER}")

            # Pose detection
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose()
            image = cv2.imread(person_path)
            if image is None:
                return "No image detected."
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            if not results.pose_landmarks:
                return "No pose detected."
            height, width, _ = image.shape
            landmarks = results.pose_landmarks.landmark
            left_shoulder = (int(landmarks[11].x * width), int(landmarks[11].y * height))
            right_shoulder = (int(landmarks[12].x * width), int(landmarks[12].y * height))
            print(f"[INFO] Shoulder coordinates: {left_shoulder}, {right_shoulder}")

            # SAM model inference
            img = load_image(person_path)
            new_tshirt = load_image(tshirt_path)
            input_points = [[[left_shoulder[0], left_shoulder[1]], [right_shoulder[0], right_shoulder[1]]]]
            inputs = processor(img, input_points=input_points, return_tensors="pt")
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

            # Serve via dynamic route
            return render_template('index.html', result_img='/outputs/result.jpg')

        except Exception as e:
            print(f"[ERROR] {e}")
            return f"Error: {e}"

    return render_template('index.html')

if __name__ == '__main__':
        
    print("[INFO] Starting Flask server...")
    app.run(debug=True, host='0.0.0.0')
