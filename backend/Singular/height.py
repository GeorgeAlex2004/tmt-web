import cv2
import torch
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import time
import os

# --- Configuration ---
YOLO_MODEL_PATH = r'D:\Work\Projects\TATA\Rib test\Training\Weights\20-08.pt'
SAM_MODEL_PATH = r"D:\Work\Projects\TATA\TATA TMT BAR ANALYZER\backend\sam_vit_h_4b8939.pth"
IMAGE_PATH = r"C:\Users\georg\OneDrive\Desktop\test1.jpg"
KNOWN_BAR_DIAMETER_MM = 12.0
SAM_MODEL_TYPE = "vit_h"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROBUST_PEAK_N_POINTS = 5

def get_sam_mask(image_rgb, box, predictor):
    """Generates a segmentation mask for a single given box using SAM."""
    # Note: set_image is called once outside the main loop for efficiency
    input_box = np.array(box)
    masks, scores, _ = predictor.predict(box=input_box[None, :], multimask_output=False)
    return masks[0], scores[0]

def get_area_based_calibration(mask, known_diameter):
    """Calculates a perspective-robust calibration factor from the bar's core mask."""
    print("  - Performing robust area-based calibration...")
    mask_uint8 = mask.astype(np.uint8) * 255
    
    y_coords, _ = np.where(mask)
    if len(y_coords) == 0: raise ValueError("Cannot process an empty SAM mask.")
    total_mask_height = np.max(y_coords) - np.min(y_coords)
    
    kernel_size = int(total_mask_height * 0.15)
    kernel_size = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    
    kernel = np.ones((kernel_size, 1), np.uint8)
    core_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=1)
    
    core_area_px = cv2.countNonZero(core_mask)
    if core_area_px == 0:
        raise ValueError("Core mask was completely erased by morphological opening.")
        
    contours, _ = cv2.findContours(core_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Could not find contour of the bar's core after removing ribs.")
        
    core_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(core_contour)
    
    core_length_px = max(rect[1])
    if core_length_px == 0:
        raise ValueError("Measured core length is zero.")
        
    pixel_diameter = core_area_px / core_length_px
    mm_per_pixel = known_diameter / pixel_diameter
    print(f"    - Derived Pixel Diameter (Area/Length): {pixel_diameter:.2f} px")
    print(f"    - Accurate Calibration Ratio: {mm_per_pixel:.4f} mm/pixel")
    
    box_points = cv2.boxPoints(rect)
    box_points = sorted(box_points, key=lambda p: p[1])
    baseline_pt1 = tuple(box_points[0])
    baseline_pt2 = tuple(box_points[1])
    
    return mm_per_pixel, (baseline_pt1, baseline_pt2)

def measure_rib_heights(image, rib_bboxes_xyxy, mm_per_pixel, baseline, predictor):
    """Measures rib height using SAM for segmentation."""
    print(f"\n[Step 3: Processing {len(rib_bboxes_xyxy)} Detected Ribs using SAM...]")
    measurements_mm = []
    
    p1, p2 = baseline
    A = p2[1] - p1[1]
    B = p1[0] - p2[0]
    C = -A * p1[0] - B * p1[1]
    line_norm = np.sqrt(A**2 + B**2)
    if line_norm == 0: return image, []

    processed_image = image.copy()
    cv2.line(processed_image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 2)
    
    for i, rib_box in enumerate(rib_bboxes_xyxy):
        # --- NEW: Use SAM to get a precise mask for each rib ---
        rib_mask, score = get_sam_mask(None, rib_box, predictor)
        
        # The rib_box is for the whole image, so we don't need an offset
        contours, _ = cv2.findContours(rib_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours: continue
        
        rib_contour = max(contours, key=cv2.contourArea)
        
        sorted_contour_points = sorted(rib_contour, key=lambda p: p[0][1])
        top_points = sorted_contour_points[:ROBUST_PEAK_N_POINTS]
        if not top_points: continue
        
        avg_x = np.mean([p[0][0] for p in top_points])
        avg_y = np.mean([p[0][1] for p in top_points])
        peak_point_absolute = (avg_x, avg_y)
        
        distance_pixels = abs(A * peak_point_absolute[0] + B * peak_point_absolute[1] + C) / line_norm
        
        avg_baseline_y = (p1[1] + p2[1]) / 2
        if peak_point_absolute[1] < avg_baseline_y + 10:
            if distance_pixels > 1:
                depth_mm = distance_pixels * mm_per_pixel
                measurements_mm.append(depth_mm)
                # Drawing data
                text = f"{depth_mm:.2f} mm"
                text_pos = (int(rib_box[0]), int(rib_box[1]) - 10)
                cv2.drawContours(processed_image, [rib_contour], -1, (0, 0, 255), 2)
                cv2.circle(processed_image, (int(avg_x), int(avg_y)), 5, (0, 255, 255), -1)
                cv2.putText(processed_image, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return processed_image, measurements_mm

if __name__ == '__main__':
    start_time = time.time()
    print("Script started.")
    print(f"Using device: {DEVICE}")

    # --- Load Models and Image ---
    print("\n[INFO] Loading models and image...")
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_MODEL_PATH)
        sam.to(device=DEVICE)
        sam_predictor = SamPredictor(sam)
        print("  - YOLO and SAM models loaded successfully.")
        
        image_bgr = cv2.imread(IMAGE_PATH)
        if image_bgr is None: raise FileNotFoundError(f"Image not found at path: {IMAGE_PATH}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        print(f"  - Image loaded successfully. Shape: {image_rgb.shape}")
    except Exception as e:
        print(f"FATAL ERROR loading assets: {e}")
        exit()

    # --- Step 1: YOLOv8 Detection ---
    print("\n[Step 1: Running YOLOv8 detection...]")
    results = yolo_model(image_rgb, verbose=False)[0]
    bar_box_xyxy = None
    rib_bboxes_xyxy = []
    for box in results.boxes:
        class_id = int(box.cls[0])
        if class_id == 0 and bar_box_xyxy is None:
            bar_box_xyxy = box.xyxy[0].cpu().numpy()
        elif class_id == 1:
            rib_bboxes_xyxy.append(box.xyxy[0].cpu().numpy())
    
    if bar_box_xyxy is None:
        print("CRITICAL: YOLO did not detect the main TMT bar (Class ID 0). Aborting.")
        exit()
    print(f"  - Detected main bar and {len(rib_bboxes_xyxy)} ribs.")

    # --- Step 2: SAM Segmentation and Calibration ---
    print("\n[Step 2: Performing Calibration using SAM...]")
    try:
        # Set image in predictor once for efficiency
        sam_predictor.set_image(image_rgb)
        bar_mask, _ = get_sam_mask(image_rgb, bar_box_xyxy, sam_predictor)
        mm_per_pixel, baseline = get_area_based_calibration(bar_mask, KNOWN_BAR_DIAMETER_MM)
    except Exception as e:
        print(f"FATAL ERROR during calibration: {e}")
        exit()
        
    # --- Step 3: Measure Rib Heights ---
    visualized_image, measured_depths = measure_rib_heights(image_bgr, rib_bboxes_xyxy, mm_per_pixel, baseline, sam_predictor)

    # --- Final Report ---
    print("\n" + "="*50)
    print("--- Final Rib Height Measurement Results ---")
    print("="*50)
    
    if measured_depths:
        if len(measured_depths) >= 4:
            q1 = np.percentile(measured_depths, 25)
            q3 = np.percentile(measured_depths, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            filtered_depths = [d for d in measured_depths if lower_bound <= d <= upper_bound]
            print(f"  - Initial measurements found: {len(measured_depths)}")
            print(f"  - Outliers removed: {len(measured_depths) - len(filtered_depths)}")
            
            avg_depth = np.mean(filtered_depths)
            print(f"\n  - Final Average Rib Height (Outliers Removed): {avg_depth:.3f} mm")
        else:
            avg_depth = np.mean(measured_depths)
            print(f"\n  - Average Rib Height: {avg_depth:.3f} mm")
    else:
        print("\nNo ribs were measured.")

    total_time = time.time() - start_time
    print(f"\nScript finished in {total_time:.2f} seconds.")

    (h, w) = visualized_image.shape[:2]
    max_height = 900
    if h > max_height:
        ratio = max_height / float(h)
        dim = (int(w * ratio), max_height)
        display_img = cv2.resize(visualized_image, dim, interpolation=cv2.INTER_AREA)
    else:
        display_img = visualized_image

    cv2.imshow("TMT Rib Height Measurement", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()