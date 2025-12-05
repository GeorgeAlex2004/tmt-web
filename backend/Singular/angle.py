import cv2
import numpy as np
from ultralytics import YOLO
import math
import os

# --- CONFIGURATION ---
MODEL_PATH = r'D:\Work\Projects\TATA\Rib test\Training\Weights\20-08.pt'
IMAGE_PATH = r'D:\Work\Projects\TATA\Rib test\Test\WhatsApp Image 2025-08-24 at 21.26.54_6635c4c8.jpg'
OUTPUT_DIR = r'D:\Work\Projects\TATA\Rib test\Test\predict'

# Set DEBUG_MODE to False for a clean final output
DEBUG_MODE = False
TMT_BAR_CLASS_NAME = 'TMT Bar'
RIB_CLASS_NAME = 'ribs'

# --- DETECTION AND PROCESSING PARAMETERS ---
CONFIDENCE_THRESHOLD = 0.25 # Keep this low to ensure we detect as many ribs as possible

# --- Final Optimized Parameters from our most successful version ---
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150
HOUGH_THRESHOLD = 15
HOUGH_MIN_LINE_LENGTH = 18
HOUGH_MAX_LINE_GAP = 7
RIB_ANGLE_MIN = 60.0 
RIB_ANGLE_MAX = 78.0


def get_object_orientation(roi):
    """Calculates the orientation of the bar using the Hough Transform."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    
    if lines is None:
        if DEBUG_MODE: print("[DEBUG] get_object_orientation: No Hough lines found for the bar.")
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if abs(angle) < 20 or abs(angle - 180) < 20 or abs(angle + 180) < 20:
            angles.append(angle if angle >= 0 else angle + 180)
    
    if not angles:
        if DEBUG_MODE: print("[DEBUG] get_object_orientation: No near-horizontal lines found.")
        return 0.0

    median_angle = np.median(angles)
    if DEBUG_MODE: print(f"[DEBUG] get_object_orientation: Median angle of bar lines is {median_angle:.2f}")
    return median_angle

def calculate_rib_angle(roi, image_to_draw_on, roi_offset, rib_num):
    """
    DEFINITIVE METHOD: Filtered Hough Transform
    Directly finds all lines and keeps only those with a plausible rib angle.
    """
    if DEBUG_MODE: print(f"\n  [Rib #{rib_num}] ROI Shape: {roi.shape}")
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(blurred, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)

    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH, maxLineGap=HOUGH_MAX_LINE_GAP
    )
    
    if lines is None:
        if DEBUG_MODE: print(f"  [Rib #{rib_num}] Hough Transform found NO lines.")
        return None

    valid_lines = []
    if DEBUG_MODE: print(f"  [Rib #{rib_num}] Hough found {len(lines)} lines. Filtering by angle...")

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle_deg = math.degrees(math.atan2(y2 - y1, x2 - x1))
        
        if RIB_ANGLE_MIN < abs(angle_deg) < RIB_ANGLE_MAX:
            valid_lines.append(line)
            abs_p1 = (x1 + roi_offset[0], y1 + roi_offset[1])
            abs_p2 = (x2 + roi_offset[0], y2 + roi_offset[1])
            cv2.line(image_to_draw_on, abs_p1, abs_p2, (0, 255, 0), 2)
        else:
            if DEBUG_MODE:
                abs_p1 = (x1 + roi_offset[0], y1 + roi_offset[1])
                abs_p2 = (x2 + roi_offset[0], y2 + roi_offset[1])
                cv2.line(image_to_draw_on, abs_p1, abs_p2, (0, 0, 255), 1)

    if not valid_lines:
        if DEBUG_MODE: print(f"  [Rib #{rib_num}] No lines passed the angle filter.")
        return None

    angles = [math.degrees(math.atan2(line[0][3] - line[0][1], line[0][2] - line[0][0])) for line in valid_lines]
    median_angle = np.median(angles)
    
    if DEBUG_MODE: print(f"  [Rib #{rib_num}] Found {len(valid_lines)} valid lines. Raw Median Angle: {median_angle:.2f}")
    
    return median_angle

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    model = YOLO(MODEL_PATH)
    image = cv2.imread(IMAGE_PATH)
    
    if image is None: print(f"Error: Could not load image at {IMAGE_PATH}"); exit()
        
    output_image = image.copy()
    results = model(image)
    
    main_bar_angle = None
    all_boxes = []
    
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            all_boxes.append((box, class_name))
            
            if main_bar_angle is None and class_name == TMT_BAR_CLASS_NAME and box.conf[0] > CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                main_bar_angle = get_object_orientation(image[y1:y2, x1:x2])
                if main_bar_angle is not None:
                    print(f"Detected main '{TMT_BAR_CLASS_NAME}' with orientation: {main_bar_angle:.2f} degrees.")
            
    if main_bar_angle is None:
        print(f"Error: Could not detect the main '{TMT_BAR_CLASS_NAME}'. Assuming 0.0 degrees.")
        main_bar_angle = 0.0

    print("\n--- Analyzing Ribs ---")
    rib_count = 0
    all_final_angles = []
    
    for box, class_name in all_boxes:
        if class_name == RIB_CLASS_NAME and box.conf[0] > CONFIDENCE_THRESHOLD:
            rib_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if DEBUG_MODE: print(f"\n-> Processing Rib #{rib_count} at BBox: [{x1}, {y1}, {x2}, {y2}]")
            
            object_roi = image[y1:y2, x1:x2]
            raw_rib_angle = calculate_rib_angle(object_roi, output_image, (x1, y1), rib_count)
            
            result_text = f"Rib #{rib_count}: "
            if raw_rib_angle is not None:
                bar_norm = main_bar_angle % 180
                rib_norm = raw_rib_angle % 180
                
                delta = abs(rib_norm - bar_norm)
                final_angle = min(delta, 180 - delta)
                
                all_final_angles.append(final_angle)
                result_text += f"{final_angle:.2f} deg"
                
                if DEBUG_MODE:
                    print(f"  [Rib #{rib_count}] Calculation: delta = abs({rib_norm:.2f} - {bar_norm:.2f}) = {delta:.2f}")
                    print(f"  [Rib #{rib_count}] Final Angle = {final_angle:.2f} degrees")
            else:
                result_text += "N/A"
                if DEBUG_MODE: print(f"  [Rib #{rib_count}] Final Angle = N/A")
            
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
            cv2.putText(output_image, result_text, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    if len(all_final_angles) > 4:
        if DEBUG_MODE: print(f"\n--- Final Calculation --- \n[DEBUG] All collected angles: {[round(a, 2) for a in all_final_angles]}")
        
        q1 = np.percentile(all_final_angles, 25)
        q3 = np.percentile(all_final_angles, 75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        
        if DEBUG_MODE:
            print(f"[DEBUG] IQR outlier rejection: Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}")
            print(f"[DEBUG] Valid angle range: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        inlier_angles = [angle for angle in all_final_angles if lower_bound <= angle <= upper_bound]
        
        if DEBUG_MODE: print(f"[DEBUG] Inlier angles after filtering: {[round(a, 2) for a in inlier_angles]}")

        if inlier_angles:
            robust_mean_angle = np.mean(inlier_angles)
            
            print("\n-------------------------------------------")
            print(f"✅ FINAL ROBUST ANGLE: {robust_mean_angle:.2f} degrees")
            print("-------------------------------------------")
            
            final_text = f"Overall Angle: {robust_mean_angle:.2f} deg"
            cv2.putText(output_image, final_text, (15, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            print("\nCould not determine a robust final angle after outlier removal.")
    else:
        print("\nNot enough valid rib angles detected to calculate a final robust angle.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_filename = os.path.basename(IMAGE_PATH)
    filename, ext = os.path.splitext(base_filename)
    output_filename = f"{filename}_predicted{ext}"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    cv2.imwrite(output_path, output_image)
    print(f"\n✅ Prediction saved to: {output_path}")
    
    cv2.imshow('Filtered Hough Transform', output_image)
    print("\nPress any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()