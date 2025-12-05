import cv2
import torch
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
# MODIFIED IMPORT: Added peak_widths
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt
import traceback
import scipy

# --- Diagnostic Print ---
print(f"--- Using SciPy Version: {scipy.__version__} ---")

# --- Configuration ---
YOLO_MODEL_PATH = r'D:\Work\Projects\TATA\Rib test\Training\Weights\20-08.pt'
SAM_MODEL_PATH = r"D:\Work\Projects\TATA\TATA TMT BAR ANALYZER\backend\sam_vit_h_4b8939.pth"
IMAGE_PATH = r'D:\Work\Projects\TATA\Rib test\Test\latest.jpg'
KNOWN_DIAMETER_MM = 12.0
SAM_MODEL_TYPE = "vit_h"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_sam_mask(image_rgb, box):
    """Generates a precise segmentation mask for the bar using SAM."""
    try:
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_MODEL_PATH)
        sam.to(device=DEVICE)
        predictor = SamPredictor(sam)
        predictor.set_image(image_rgb)
        masks, _, _ = predictor.predict(box=np.array(box)[None, :], multimask_output=False)
        return masks[0]
    except Exception as e:
        print(f"[ERROR] in get_sam_mask: {e}")
        raise

def get_calibration_and_rect(mask, known_diameter):
    """Calculates the mm/pixel ratio and finds the bar's orientation."""
    try:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: raise ValueError("No contours found in SAM mask.")
        bar_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(bar_contour)
        pixel_diameter = min(rect[1])
        if pixel_diameter == 0: raise ValueError("Measured pixel diameter is zero.")
        mm_per_pixel = known_diameter / pixel_diameter
        print(f"[INFO] Calibration factor: {mm_per_pixel:.4f} mm/pixel")
        return mm_per_pixel, rect
    except Exception as e:
        print(f"[ERROR] in get_calibration_and_rect: {e}")
        raise

def get_intensity_profile(image, mask, rect):
    """Rotates the bar to be horizontal and creates a 1D intensity profile."""
    try:
        masked_image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))
        gray_masked = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        center, (width, height), angle = rect
        if width < height:
            angle += 90
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        h, w = gray_masked.shape
        rotated_gray = cv2.warpAffine(gray_masked, M, (w, h))
        box_pts = np.intp(cv2.boxPoints(rect))
        pts = np.intp(cv2.transform(np.array([box_pts]), M)[0])
        crop_x_start, crop_x_end = np.min(pts[:, 0]), np.max(pts[:, 0])
        crop_y_start, crop_y_end = np.min(pts[:, 1]), np.max(pts[:, 1])
        bar_crop = rotated_gray[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
        if bar_crop.size == 0: raise ValueError("Cropped bar area is empty.")
        return np.mean(bar_crop, axis=0)
    except Exception as e:
        print(f"[ERROR] in get_intensity_profile: {e}")
        raise

# ==============================================================================
# REPLACED FUNCTION: This version now uses the more compatible two-step method.
# ==============================================================================
def measure_transverse_rib_length(intensity_profile, mm_per_pixel):
    """Analyzes the intensity profile to find the average rib length (Ltr)."""
    try:
        # Step 1: Find the peaks without the width argument
        peaks, _ = find_peaks(
            intensity_profile,
            prominence=5,
            distance=10
        )

        if len(peaks) == 0:
            print("[WARNING] No ribs were detected.")
            return 0.0, None, None

        # Step 2: Calculate the widths of the found peaks separately using peak_widths
        widths_info = peak_widths(
            intensity_profile,
            peaks,
            rel_height=0.5
        )
        pixel_widths = widths_info[0]

        mm_widths = pixel_widths * mm_per_pixel

        if len(mm_widths) > 3:
            median_width = np.median(mm_widths)
            final_widths = [w for w in mm_widths if abs(w - median_width) / median_width < 0.35]
        else:
            final_widths = mm_widths

        avg_ltr_mm = np.mean(final_widths) if len(final_widths) > 0 else 0.0

        # Create a 'properties' dictionary compatible with the visualization function
        props = {
            "width_heights": widths_info[1],
            "left_ips": widths_info[2],
            "right_ips": widths_info[3]
        }

        return avg_ltr_mm, peaks, props
    except Exception as e:
        print(f"[ERROR] in measure_transverse_rib_length: {e}")
        raise

def visualize_results(profile, peaks, properties):
    """Plots the intensity profile and the measured widths of the peaks."""
    plt.figure(figsize=(15, 6))
    plt.plot(profile, label='Intensity Profile')
    plt.plot(peaks, profile[peaks], "x", c='r', label='Detected Ribs (Peaks)')

    plt.hlines(
        y=properties["width_heights"],
        xmin=properties["left_ips"],
        xmax=properties["right_ips"],
        color="limegreen",
        linewidth=3,
        label='Measured Rib Width (Ltr)'
    )

    plt.title("Rib Intensity Profile and Measured Lengths")
    plt.xlabel("Position along Bar (pixels)")
    plt.ylabel("Average Pixel Intensity")
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        image = cv2.imread(IMAGE_PATH)
        if image is None: raise FileNotFoundError(f"Image not found at {IMAGE_PATH}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = yolo_model(image_rgb)
        box = results[0].boxes.xyxy[0].cpu().numpy()
        sam_mask = get_sam_mask(image_rgb, box)
        mm_per_pixel, bar_rect = get_calibration_and_rect(sam_mask, KNOWN_DIAMETER_MM)
        profile = get_intensity_profile(image, sam_mask, bar_rect)
        avg_ltr, peaks, props = measure_transverse_rib_length(profile, mm_per_pixel)

        print("\n" + "="*40)
        print("    Transverse Rib Length (Ltr) Result")
        print("="*40)
        print(f"Average Rib Length: {avg_ltr:.3f} mm")
        print("="*40)

        if props:
            visualize_results(profile, peaks, props)
    except Exception as e:
        print(f"\n[FATAL ERROR] An error occurred: {e}")
        traceback.print_exc()