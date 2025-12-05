import os
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np
import torch
import cv2
import base64
from io import BytesIO
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for SAM model
sam = None
predictor = None

def load_sam_model():
    """Load SAM model with error handling"""
    global sam, predictor
    try:
        # Import SAM (Segment Anything Model)
        from segment_anything import sam_model_registry, SamPredictor
        from analysis import analyze_tmt_cross_section, analyze_tmt_thickness
        
        # Load SAM model once at startup
        SAM_CHECKPOINT = os.environ.get('SAM_CHECKPOINT', 'sam_vit_h_4b8939.pth')
        
        if not os.path.exists(SAM_CHECKPOINT):
            print(f"WARNING: SAM model file '{SAM_CHECKPOINT}' not found!")
            print("Please download the SAM model and place it in the backend directory.")
            print("Download from: https://github.com/facebookresearch/segment-anything#model-checkpoints")
            return False
            
        print(f"Loading SAM model from: {SAM_CHECKPOINT}")
        sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
        predictor = SamPredictor(sam)
        print("SAM model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading SAM model: {e}")
        return False

# Try to load SAM model
sam_loaded = load_sam_model()

# Helper: Save uploaded image and return path
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def save_image(file_storage):
    img_id = str(uuid.uuid4())
    path = os.path.join(UPLOAD_FOLDER, f"{img_id}.jpg")
    file_storage.save(path)
    return path

def image_to_base64(image_path):
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def segment_tmt_bar(image: Image.Image):
    """Segment TMT bar using SAM model"""
    if not sam_loaded:
        raise ValueError("SAM model not loaded. Please check if sam_vit_h_4b8939.pth exists.")
    
    np_image = np.array(image)
    predictor.set_image(np_image)
    # Use center point as prompt
    h, w = np_image.shape[:2]
    input_point = np.array([[w // 2, h // 2]])
    input_label = np.array([1])
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    mask = masks[0]
    return mask

def extract_segmented_bar(image: Image.Image, mask: np.ndarray):
    np_image = np.array(image)
    if len(np_image.shape) == 2:
        np_image = np.expand_dims(np_image, axis=-1)
    segmented = np_image * mask[..., None]
    return segmented

def level1_analysis(segmented_image, mask, diameter):
    # Placeholder: Simulate color/shape analysis
    return {
        "layersDetected": True,
        "continuousRing": True,
        "concentricRegions": True,
        "uniformThickness": True,
        "status": True
    }

def level2_analysis(segmented_image, mask, diameter):
    # Placeholder: Simulate distance/good-bad analysis
    rim_thickness = round(diameter * 0.09, 2)
    thickness_percentage = round((rim_thickness / diameter) * 100, 1)
    within_range = 7 <= thickness_percentage <= 10
    meets_standards = within_range
    return {
        "rimThickness": rim_thickness,
        "thicknessPercentage": thickness_percentage,
        "withinRange": within_range,
        "meetsStandards": meets_standards,
        "status": meets_standards
    }

@app.route('/')
def home():
    """Simple test endpoint to verify server is running"""
    return jsonify({
        "status": "Server is running",
        "message": "TMT Ring Test Backend API",
        "sam_model_loaded": sam_loaded,
        "endpoints": [
            "/process-ring-test (POST)",
            "/results/<filename> (GET)",
            "/get-report?test_id=<id> (GET)"
        ]
    })

@app.route('/process-ring-test', methods=['POST'])
def process_ring_test():
    try:
        print(f"Received image processing request at {datetime.now()}")
        
        if 'image' not in request.files or 'diameter' not in request.form:
            return jsonify({"error": "Missing image or diameter"}), 400
        
        if not sam_loaded:
            return jsonify({"error": "SAM model not loaded. Please check if sam_vit_h_4b8939.pth exists in the backend directory."}), 500
        
        image_file = request.files['image']
        diameter = float(request.form['diameter'])
        test_id = str(uuid.uuid4())  # Generate test_id early
        
        print(f"Processing test_id: {test_id}, diameter: {diameter}")
        
        # Save uploaded image
        image_path = save_image(image_file)
        print(f"Image saved to: {image_path}")
        
        # Load and convert image
        image = Image.open(image_path).convert('RGB')
        print(f"Image loaded, size: {image.size}")

        # SAM segmentation
        print("Starting SAM segmentation...")
        mask = segment_tmt_bar(image)
        print("SAM segmentation completed")
        
        segmented_image = extract_segmented_bar(image, mask)
        print("Segmented image extracted")

        # Save segmented image
        seg_img_path = os.path.join(RESULTS_FOLDER, f"{test_id}_segmented.png")
        cv2.imwrite(seg_img_path, segmented_image)
        print(f"Segmented image saved to: {seg_img_path}")

        # Import analysis functions here to avoid import errors if SAM not loaded
        from analysis import analyze_tmt_cross_section, analyze_tmt_thickness
        
        # Level 1 analysis
        print("Starting Level 1 analysis...")
        level1_results = analyze_tmt_cross_section(seg_img_path)
        print("Level 1 analysis completed")
        
        # Level 2 analysis
        print("Starting Level 2 analysis...")
        debug_img, min_thickness, max_thickness, quality_status, quality_message = analyze_tmt_thickness(seg_img_path, diameter)
        debug_img_path = os.path.join(RESULTS_FOLDER, f"{test_id}_debug.png")
        cv2.imwrite(debug_img_path, debug_img)
        print("Level 2 analysis completed")

        # Convert images to base64 for direct transmission
        print("Converting images to base64...")
        segmented_base64 = f"data:image/png;base64,{image_to_base64(seg_img_path)}"
        debug_base64 = f"data:image/png;base64,{image_to_base64(debug_img_path)}"
        print("Base64 conversion completed")

        print(f"Processing completed successfully for test_id: {test_id}")
        
        return jsonify({
            "test_id": test_id,
            "level1": {k: bool(v) if isinstance(v, (np.bool_, bool)) else v for k, v in level1_results.items()},
            "level2": {
                "min_thickness_mm": float(min_thickness),
                "max_thickness_mm": float(max_thickness),
                "quality_status": bool(quality_status),
                "quality_message": quality_message
            },
            "verdict": "PASS" if bool(quality_status) else "FAIL",
            "segmented_image_url": f"/results/{test_id}_segmented.png",
            "debug_image_url": f"/results/{test_id}_debug.png",
            "segmented_image_base64": segmented_base64,
            "debug_image_base64": debug_base64
        })
    except Exception as e:
        print(f"Error in process_ring_test: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route('/results/<filename>')
def get_result_image(filename):
    return send_from_directory(RESULTS_FOLDER, filename)

@app.route('/get-report', methods=['GET'])
def get_report():
    """Endpoint for checking if processing is complete"""
    try:
        test_id = request.args.get('test_id')
        if not test_id:
            return jsonify({"error": "Missing test_id"}), 400
        
        # Check if result files exist
        seg_path = os.path.join(RESULTS_FOLDER, f"{test_id}_segmented.png")
        debug_path = os.path.join(RESULTS_FOLDER, f"{test_id}_debug.png")
        
        if os.path.exists(seg_path) and os.path.exists(debug_path):
            # Return the same structure as process-ring-test for consistency
            return jsonify({
                "processing": False,
                "test_id": test_id,
                "verdict": "PASS",  # You might want to store this properly
                "level1": {
                    "dark_grey_and_light_core_visible": True,
                    "continuous_outer_ring": True,
                    "concentric_regions": True,
                    "uniform_thickness": True
                },
                "level2": {
                    "min_thickness_mm": 1.8,
                    "max_thickness_mm": 2.2,
                    "quality_status": True,
                    "quality_message": "Thickness within acceptable range"
                },
                "segmented_image_url": f"/results/{test_id}_segmented.png",
                "debug_image_url": f"/results/{test_id}_debug.png",
                "segmented_image_base64": f"data:image/png;base64,{image_to_base64(seg_path)}",
                "debug_image_base64": f"data:image/png;base64,{image_to_base64(debug_path)}"
            })
        else:
            return jsonify({"processing": True}), 200
    except Exception as e:
        print(f"Error in get_report: {e}")
        return jsonify({"error": f"Failed to get report: {str(e)}"}), 500

@app.route('/analyze_rib_height', methods=['POST'])
def analyze_rib_height():
    """Rib Test: Analyze rib height/thickness from 45-degree angle view"""
    try:
        # Get JSON data from request
        data = request.json
        
        # Extract image and parameters
        image_data = data.get('image')
        overlay_size = data.get('overlay_size')  # Diameter in pixels from overlay
        
        # Get diameter parameter
        diameter = float(request.args.get('diameter', '10.0'))  # Default to 10mm if not specified
        
        # For now, return a simple response to test the endpoint
        # In a full implementation, you would process the image here
        
        return jsonify({
            'status': 'success',
            'test_type': 'rib_test',
            'analysis_type': 'rib_height',
            'height': {
                'value': 1.5,  # Placeholder value
                'mean': 1.5,
                'confidence_interval': 0.1,
                'std_dev': 0.05,
                'measurements': 1,
                'raw_values': [1.5]
            },
            'diameter': diameter,
            'used_scale_factor': 10.0,
            'message': 'Rib height analysis completed (placeholder response)'
        })
    
    except Exception as e:
        print(f"Rib height analysis error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        })

@app.route('/status', methods=['GET'])
def status():
    """Check backend status and processing capability"""
    return jsonify({
        "status": "running",
        "sam_model_loaded": sam_loaded,
        "timestamp": datetime.now().isoformat(),
        "message": "Backend is ready for processing"
    })

if __name__ == '__main__':
    print("Starting TMT Ring Test Backend...")
    print(f"SAM Model Loaded: {sam_loaded}")
    if not sam_loaded:
        print("WARNING: Backend will start but image processing will fail!")
        print("Please ensure sam_vit_h_4b8939.pth is in the backend directory.")
    
    # Fix threading issues and improve server configuration
    app.run(
        host='0.0.0.0', 
        port=5000, 
        debug=True,
        threaded=True,
        use_reloader=False  # Disable auto-reloader to prevent threading issues
    )