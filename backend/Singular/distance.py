import cv2
import torch
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import traceback # Used for detailed error reporting

# --- Configuration ---
YOLO_MODEL_PATH = r'D:\Work\Projects\TATA\Rib test\Training\Weights\20-08.pt'
SAM_MODEL_PATH = r"D:\Work\Projects\TATA\TATA TMT BAR ANALYZER\backend\sam_vit_h_4b8939.pth"
IMAGE_PATH = r'D:\Work\Projects\TATA\Rib test\Test\test1.jpg'
KNOWN_DIAMETER_MM = 12.0
SAM_MODEL_TYPE = "vit_h"

# Set the device to use for computation (CUDA if available, otherwise CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_sam_mask(image_rgb, box):
    """Generates a segmentation mask for the given box using SAM."""
    print("\n[DEBUG] --- Entering get_sam_mask function ---")
    print(f"[DEBUG] Input box (xyxy): {box}")
    
    try:
        print("[DEBUG] Initializing SAM model registry...")
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_MODEL_PATH)
        sam.to(device=DEVICE)
        predictor = SamPredictor(sam)
        print("[DEBUG] SAM model loaded and predictor created.")
        
        print("[DEBUG] Setting image for SAM predictor...")
        predictor.set_image(image_rgb)
        print("[DEBUG] Image set successfully.")
        
        input_box = np.array(box)
        
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        
        print(f"[DEBUG] SAM prediction complete. Found {len(masks)} mask(s).")
        print(f"[DEBUG] Mask scores: {scores}")
        print(f"[DEBUG] Selected mask shape: {masks[0].shape}, dtype: {masks[0].dtype}")
        print("[DEBUG] --- Exiting get_sam_mask function ---")
        return masks[0]
    except Exception as e:
        print(f"[ERROR] Exception in get_sam_mask: {e}")
        traceback.print_exc()
        raise

def calculate_calibration_and_roi(mask, known_diameter):
    """
    Calculates mm_per_pixel ratio using the mask's width and finds the
    minimum area rectangle for robust analysis.
    """
    print("\n[DEBUG] --- Entering calculate_calibration_and_roi function ---")
    print(f"[DEBUG] Input mask shape: {mask.shape}, non-zero pixels: {np.count_nonzero(mask)}")
    
    try:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"[DEBUG] Found {len(contours)} contours in the mask.")
        
        if not contours:
            raise ValueError("Could not find contours in the SAM mask.")
            
        bar_contour = max(contours, key=cv2.contourArea)
        print(f"[DEBUG] Largest contour selected with area: {cv2.contourArea(bar_contour)}")
        
        rect = cv2.minAreaRect(bar_contour)
        print(f"[DEBUG] minAreaRect calculated: center={rect[0]}, size={rect[1]}, angle={rect[2]}")
        
        pixel_diameter = min(rect[1])
        print(f"[DEBUG] Measured Pixel Diameter (shorter side of rect): {pixel_diameter:.2f} pixels")
        
        if pixel_diameter == 0:
            raise ValueError("Measured pixel diameter is zero. Cannot calibrate.")
            
        mm_per_pixel = known_diameter / pixel_diameter
        print(f"[DEBUG] Calculated Calibration Factor: {mm_per_pixel:.4f} mm/pixel")
        print("[DEBUG] --- Exiting calculate_calibration_and_roi function ---")
        
        return mm_per_pixel, rect
    except Exception as e:
        print(f"[ERROR] Exception in calculate_calibration_and_roi: {e}")
        traceback.print_exc()
        raise

def analyze_ribs(image, mask, rect, mm_per_pixel):
    """
    Analyzes ribs, uses statistical filtering, and returns synchronized lists of
    peak pairs and distances for accurate drawing.
    """
    print("\n[DEBUG] --- Entering analyze_ribs function ---")
    try:
        masked_image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))
        gray_masked = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        center, (width, height), angle = rect
        if width < height:
            angle += 90
            width, height = height, width
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        h, w = gray_masked.shape
        rotated_gray = cv2.warpAffine(gray_masked, M, (w, h))
        box = cv2.boxPoints(rect)
        pts = np.intp(cv2.transform(np.array([box]), M)[0])
        crop_x_start, crop_x_end = np.min(pts[:, 0]), np.max(pts[:, 0])
        crop_y_start, crop_y_end = np.min(pts[:, 1]), np.max(pts[:, 1])
        bar_crop = rotated_gray[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
        if bar_crop.size == 0:
            raise ValueError("Cropped bar area is empty.")
        intensity_profile = np.mean(bar_crop, axis=0)

        # --- Peak Finding ---
        peak_prominence = 5
        peak_distance = 10
        print(f"[DEBUG] Finding peaks with prominence={peak_prominence} and distance={peak_distance}")
        peaks, _ = find_peaks(intensity_profile, prominence=peak_prominence, distance=peak_distance)
        print(f"[DEBUG] Found {len(peaks)} peaks at pixel locations: {peaks}")
        
        if len(peaks) < 2:
            print("[DEBUG] Warning: Less than two ribs were detected.")
            # Return empty lists and zero values that the main function can handle
            return [], [], 0, intensity_profile, bar_crop, peaks
            
        pixel_distances = np.diff(peaks)
        mm_distances = pixel_distances * mm_per_pixel
        print(f"[DEBUG] Found {len(mm_distances)} initial distances (mm): {[f'{d:.2f}' for d in mm_distances]}")
        
        # --- STATISTICAL FILTERING AND DATA SYNCHRONIZATION ---
        final_distances = []
        final_peak_pairs = [] # This will store the (peak1, peak2) tuples that are kept
        avg_mm_distance = 0

        if len(mm_distances) > 3: # Only filter if we have a reasonable number of measurements
            median_dist = np.median(mm_distances)
            print(f"[DEBUG] Calculated median distance: {median_dist:.2f} mm")
            
            # This threshold can be adjusted. 35% deviation from median is allowed.
            filter_threshold = 0.35 

            for i, d in enumerate(mm_distances):
                if abs(d - median_dist) / median_dist < filter_threshold:
                    final_distances.append(d)
                    # Keep the pair of peaks that corresponds to this valid distance
                    final_peak_pairs.append((peaks[i], peaks[i+1]))

            print(f"[DEBUG] Filtered distances ({len(final_distances)} remaining): {[f'{d:.2f}' for d in final_distances]}")

            if len(final_distances) > 1:
                avg_mm_distance = np.mean(final_distances)
            else: # Fallback if filtering was too aggressive
                print("[DEBUG] Warning: Filtering removed too many values. Using original distances.")
                final_distances = list(mm_distances)
                final_peak_pairs = [(peaks[i], peaks[i+1]) for i in range(len(mm_distances))]
                avg_mm_distance = np.mean(final_distances)
        else:
            final_distances = list(mm_distances)
            final_peak_pairs = [(peaks[i], peaks[i+1]) for i in range(len(mm_distances))]
            if final_distances:
                avg_mm_distance = np.mean(final_distances)
        
        print(f"[DEBUG] Final average of filtered distances: {avg_mm_distance:.2f} mm")
        print("[DEBUG] --- Exiting analyze_ribs function ---")
        
        # Return the synchronized lists
        return final_distances, final_peak_pairs, avg_mm_distance, intensity_profile, bar_crop, peaks
        
    except Exception as e:
        print(f"[ERROR] Exception in analyze_ribs: {e}")
        traceback.print_exc()
        raise

def draw_results_on_image(image_rgb, mask, rect, filtered_peak_pairs, filtered_distances):
    """
    Draws annotations using the synchronized lists of peak pairs and distances.
    """
    print("\n[DEBUG] --- Entering draw_results_on_image function ---")
    try:
        display_image = image_rgb.copy()
        mask_overlay = np.zeros_like(display_image)
        mask_overlay[mask] = [0, 255, 0] # Green color for the mask
        display_image = cv2.addWeighted(display_image, 0.7, mask_overlay, 0.3, 0)
        
        center, (width, height), angle = rect
        if width < height:
            angle += 90
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        inv_M = cv2.invertAffineTransform(M)
        
        box = cv2.boxPoints(rect)
        pts = np.intp(cv2.transform(np.array([box]), M)[0])
        crop_x_start = np.min(pts[:, 0])

        # This loop iterates over the filtered data, so no IndexError can occur.
        for i in range(len(filtered_distances)):
            p1_x_crop, p2_x_crop = filtered_peak_pairs[i]
            dist_text = f"{filtered_distances[i]:.2f}mm"

            # Draw line for the first peak in the pair
            pt1_top_rot = (p1_x_crop + crop_x_start, np.min(pts[:, 1]))
            pt1_bot_rot = (p1_x_crop + crop_x_start, np.max(pts[:, 1]))
            pt1_top_orig = tuple(np.intp(cv2.transform(np.array([[pt1_top_rot]]), inv_M)[0][0]))
            pt1_bot_orig = tuple(np.intp(cv2.transform(np.array([[pt1_bot_rot]]), inv_M)[0][0]))
            cv2.line(display_image, pt1_top_orig, pt1_bot_orig, (255, 0, 0), 2) # Blue line

            # Add distance text at the midpoint
            midpoint_x_crop = (p1_x_crop + p2_x_crop) / 2
            midpoint_rot = (midpoint_x_crop + crop_x_start, center[1])
            text_pos_orig = tuple(np.intp(cv2.transform(np.array([[midpoint_rot]]), inv_M)[0][0]))
            
            (tw, th), _ = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display_image, (text_pos_orig[0] - tw//2, text_pos_orig[1] - th), (text_pos_orig[0] + tw//2, text_pos_orig[1] + 5), (255,255,255), -1)
            cv2.putText(display_image, dist_text, (text_pos_orig[0] - tw//2, text_pos_orig[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2) # Black text

        # Draw the very last peak line if it was part of a valid pair
        if filtered_peak_pairs:
            last_peak_x_crop = filtered_peak_pairs[-1][1]
            last_pt_top_rot = (last_peak_x_crop + crop_x_start, np.min(pts[:, 1]))
            last_pt_bot_rot = (last_peak_x_crop + crop_x_start, np.max(pts[:, 1]))
            last_pt_top_orig = tuple(np.intp(cv2.transform(np.array([[last_pt_top_rot]]), inv_M)[0][0]))
            last_pt_bot_orig = tuple(np.intp(cv2.transform(np.array([[last_pt_bot_rot]]), inv_M)[0][0]))
            cv2.line(display_image, last_pt_top_orig, last_pt_bot_orig, (255, 0, 0), 2)
        
        print("[DEBUG] --- Exiting draw_results_on_image function ---")
        return display_image
    except Exception as e:
        print(f"[ERROR] Exception in draw_results_on_image: {e}")
        traceback.print_exc()
        raise

def main():
    """Main function to run the entire pipeline."""
    print("[INFO] --- Starting TMT Bar Rib Analysis Script ---")
    print(f"[INFO] YOLO Model Path: {YOLO_MODEL_PATH}")
    print(f"[INFO] SAM Model Path: {SAM_MODEL_PATH}")
    print(f"[INFO] Image Path: {IMAGE_PATH}")
    print(f"[INFO] Known Bar Diameter: {KNOWN_DIAMETER_MM} mm")
    print(f"[INFO] Using device: {DEVICE}")

    try:
        # 1. Load Image
        image_bgr = cv2.imread(IMAGE_PATH)
        if image_bgr is None:
            print(f"[ERROR] Could not load image from {IMAGE_PATH}. Check file path.")
            return
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        print(f"[INFO] Image loaded successfully. Shape: {image_rgb.shape}")

        # 2. Run YOLOv8 to get bounding box, filtering by class ID
        print("\n[INFO] --- Step 1: Running YOLOv8 detection ---")
        # IMPORTANT: Change this ID if your 'tmt_bar' class is not 0
        TMT_BAR_CLASS_ID = 0 
        model = YOLO(YOLO_MODEL_PATH)
        results = model(image_rgb, verbose=False)

        bar_box = None
        # Loop through all detections to find the one for the TMT bar
        for box in results[0].boxes:
            if box.cls == TMT_BAR_CLASS_ID:
                bar_box = box.xyxy[0].cpu().numpy()
                print(f"[INFO] Found TMT bar (Class ID {TMT_BAR_CLASS_ID})")
                break # Stop after finding the first one

        if bar_box is None:
            print(f"[ERROR] YOLOv8 did not detect a TMT bar with Class ID {TMT_BAR_CLASS_ID}.")
            if len(results[0].boxes) > 0:
                detected_classes = results[0].boxes.cls.cpu().numpy()
                print(f"[DEBUG] Detected Class IDs were: {detected_classes}")
            return

        print(f"[INFO] YOLOv8 detected TMT bar. Bounding box (xyxy): {bar_box}")

        # 3. Get SAM Mask
        print("\n[INFO] --- Step 2: Generating segmentation mask with SAM ---")
        bar_mask = get_sam_mask(image_rgb, bar_box)

        # 4. Calibrate using the mask
        print("\n[INFO] --- Step 3: Calibrating pixel-to-mm ratio ---")
        mm_per_pixel, bar_rect = calculate_calibration_and_roi(bar_mask, KNOWN_DIAMETER_MM)

        # 5. Analyze Ribs
        print("\n[INFO] --- Step 4: Analyzing ribs ---")
        # The return values from analyze_ribs have changed
        filtered_distances, filtered_peak_pairs, avg_mm_dist, profile, _, peaks = analyze_ribs(image_bgr, bar_mask, bar_rect, mm_per_pixel)

        # 6. Report and Visualize Results
        print("\n[INFO] --- Step 5: Generating final report and visualization ---")
        # Pass the new synchronized lists to the drawing function
        annotated_image = draw_results_on_image(image_rgb, bar_mask, bar_rect, filtered_peak_pairs, filtered_distances)
        
        print("\n" + "="*20 + " FINAL RESULTS " + "="*20)
        print(f"Number of reliable ribs distances measured: {len(filtered_distances)}")
        if avg_mm_dist > 0:
            print(f"Final Average Rib Distance: {avg_mm_dist:.2f} mm")
            print("Reliable individual distances (mm):", [f"{d:.2f}" for d in filtered_distances])
        print("="*60)
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]})
        axes[0].imshow(annotated_image)
        axes[0].set_title("Segmented TMT Bar with Rib Distance Measurements")
        axes[0].axis('off')
        axes[1].plot(profile)
        # Plot all initially found peaks for comparison
        ax1_twin = axes[1].twinx()
        ax1_twin.plot(peaks, profile[peaks], "x", color='r', markersize=10, label=f"Detected Peaks ({len(peaks)})")
        axes[1].set_title(f"Intensity Profile Analysis (Final Average: {avg_mm_dist:.2f} mm)")
        axes[1].set_xlabel("Position along the bar (pixels)")
        axes[1].set_ylabel("Average Pixel Intensity")
        axes[1].grid(True)
        ax1_twin.legend()
        plt.tight_layout()
        print("\n[INFO] Displaying visualization. Close the plot window to exit.")
        plt.show()

    except Exception as e:
        print("\n" + "#"*20 + " AN UNHANDLED ERROR OCCURRED " + "#"*20)
        print(f"[FATAL] An error occurred in the main pipeline: {e}")
        traceback.print_exc()
        print("#"*70)

if __name__ == "__main__":
    main()