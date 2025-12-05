import cv2
import numpy as np

# Level 1: Cross-section analysis

def analyze_tmt_cross_section(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read the image")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = {
        "dark_grey_and_light_core_visible": False,
        "continuous_outer_ring": False,
        "concentric_regions": False,
        "uniform_thickness": False
    }
    if len(contours) > 0:
        main_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [main_contour], -1, (255), -1)
        outer_mean = cv2.mean(gray, mask=mask)[0]
        inner_mask = cv2.bitwise_not(mask)
        inner_mean = cv2.mean(gray, mask=inner_mask)[0]
        results["dark_grey_and_light_core_visible"] = abs(outer_mean - inner_mean) > 20
        perimeter = cv2.arcLength(main_contour, True)
        approx = cv2.approxPolyDP(main_contour, 0.02 * perimeter, True)
        results["continuous_outer_ring"] = len(approx) > 8
        (x, y), radius = cv2.minEnclosingCircle(main_contour)
        center = np.array([x, y])
        distances = np.sqrt(np.sum((main_contour[:, 0] - center) ** 2, axis=1))
        std_dev = np.std(distances)
        results["concentric_regions"] = std_dev < (np.mean(distances) * 0.1)
        thickness_variation = np.std(distances) / np.mean(distances)
        results["uniform_thickness"] = thickness_variation < 0.15
    return results

# Level 2: Thickness and quality analysis

def read_and_preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read the image. Check the file path.")
    img_enhanced = cv2.equalizeHist(img)
    blurred = cv2.GaussianBlur(img_enhanced, (5, 5), 0)
    return blurred

def find_contours(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in the image")
    return sorted(contours, key=cv2.contourArea, reverse=True)[:2]

def calculate_thickness(outer_contour, inner_contour, center, outer_radius, inner_radius, actual_diameter_mm):
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    thicknesses = []
    debug_points = []
    for angle in angles:
        outer_x = int(center[0] + outer_radius * np.cos(angle))
        outer_y = int(center[1] + outer_radius * np.sin(angle))
        inner_x = int(center[0] + inner_radius * np.cos(angle))
        inner_y = int(center[1] + inner_radius * np.sin(angle))
        thickness = np.sqrt((outer_x - inner_x) ** 2 + (outer_y - inner_y) ** 2)
        thicknesses.append(thickness)
        debug_points.append(((outer_x, outer_y), (inner_x, inner_y)))
    pixels_per_mm = outer_radius / (actual_diameter_mm / 2)
    thicknesses_mm = [t / pixels_per_mm for t in thicknesses]
    return thicknesses_mm, debug_points

def annotate_debug_image(img, contours, debug_points, thicknesses_mm, min_idx, max_idx, quality_message, is_good_quality):
    debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(debug_img, [contours[0]], -1, (0, 255, 0), 2)
    cv2.drawContours(debug_img, [contours[1]], -1, (255, 0, 0), 2)
    cv2.line(debug_img, debug_points[min_idx][0], debug_points[min_idx][1], (0, 0, 255), 2)
    cv2.line(debug_img, debug_points[max_idx][0], debug_points[max_idx][1], (255, 0, 0), 2)
    font_scale, font_thickness = 0.7, 2
    cv2.putText(debug_img, f"Min: {thicknesses_mm[min_idx]:.2f} mm", debug_points[min_idx][0], cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)
    cv2.putText(debug_img, f"Max: {thicknesses_mm[max_idx]:.2f} mm", debug_points[max_idx][0], cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), font_thickness)
    quality_color = (0, 255, 0) if is_good_quality else (0, 0, 255)
    cv2.putText(debug_img, quality_message, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, quality_color, 2)
    return debug_img

def analyze_tmt_thickness(image_path, actual_diameter_mm):
    img = read_and_preprocess_image(image_path)
    binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 2:
        # Not enough contours found, return a default/failure result
        debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        quality_message = "Could not detect both outer and inner contours."
        cv2.putText(debug_img, quality_message, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return debug_img, 0, 0, False, quality_message
    
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    if cv2.contourArea(contours[0]) > cv2.contourArea(contours[1]):
        outer_contour, inner_contour = contours[0], contours[1]
    else:
        outer_contour, inner_contour = contours[1], contours[0]
    M = cv2.moments(outer_contour)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    def calc_radius(contour, cx, cy):
        return [np.sqrt((point[0][0] - cx) ** 2 + (point[0][1] - cy) ** 2) for point in contour]
    outer_radii = calc_radius(outer_contour, *center)
    inner_radii = calc_radius(inner_contour, *center)
    outer_radius = np.mean(outer_radii)
    inner_radius = np.mean(inner_radii)
    thicknesses_mm, debug_points = calculate_thickness(outer_contour, inner_contour, center, outer_radius, inner_radius, actual_diameter_mm)
    min_thickness_mm, max_thickness_mm = min(thicknesses_mm), max(thicknesses_mm)
    min_idx, max_idx = thicknesses_mm.index(min_thickness_mm), thicknesses_mm.index(max_thickness_mm)
    min_acceptable = actual_diameter_mm * 0.07
    max_acceptable = actual_diameter_mm * 0.10
    is_good_quality = (min_thickness_mm >= min_acceptable and max_thickness_mm <= max_acceptable) if thicknesses_mm else False
    quality_message = (
        f"Good TMT Bar: Thickness ({min_thickness_mm:.2f}mm - {max_thickness_mm:.2f}mm) is within acceptable limits."
        if is_good_quality else
        f"Poor TMT Bar: Thickness ({min_thickness_mm:.2f}mm - {max_thickness_mm:.2f}mm) is outside acceptable limits."
    )
    debug_img = annotate_debug_image(img, contours, debug_points, thicknesses_mm, min_idx, max_idx, quality_message, is_good_quality)
    return debug_img, min_thickness_mm, max_thickness_mm, is_good_quality, quality_message 