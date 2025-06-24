import os
import sys
import cv2
import numpy as np
import csv
from PIL import Image
import matplotlib.pyplot as plt

def rgb_to_lab_manual(image):
    
    # Normalize the image (convert 8-bit RGB to range [0,1])
    image = image / 255.0
    
    # Convert RGB to XYZ color space
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    
    xyz_image = np.dot(image, M.T)
    
    # Normalize for the D65 illuminant
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    xyz_image[..., 0] /= Xn
    xyz_image[..., 1] /= Yn
    xyz_image[..., 2] /= Zn
    
    def f(t):
        delta = 6/29
        return np.where(t > delta**3, t**(1/3), (t / (3 * delta**2)) + (4/29))
    
    f_x = f(xyz_image[..., 0])
    f_y = f(xyz_image[..., 1])
    f_z = f(xyz_image[..., 2])
    
    # Compute LAB channels
    L = (116 * f_y) - 16
    L = np.clip(L * 255 / 100, 0, 255)  # Scale L to 8-bit range
    
    return L.astype(np.uint8)

def extract_L_channel(image_path, output_path):
    """
    Extract the L channel from an image manually.
    """
    # Load the image in BGR format
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return
    
    # Convert BGR to RGB manually
    image = np.array(image[..., ::-1], dtype=np.float32)  # Convert to NumPy array and change BGR to RGB
    # Convert to L-channel manually
    L_channel = rgb_to_lab_manual(image)
    
    # Save the L-channel image
    cv2.imwrite(output_path, L_channel)
    print(f"L-channel image saved at: {output_path}")

def generate_gaussian_kernel(kernel_size, sigma=1.0):
    kernel = []
    sum_val = 0
    center = kernel_size // 2
    for i in range(kernel_size):
        row = []
        for j in range(kernel_size):
            x, y = i - center, j - center
            value = (1 / (2 * 3.14159265359 * sigma ** 2)) * (2.71828182846 ** (-(x ** 2 + y ** 2) / (2 * sigma ** 2)))
            row.append(value)
            sum_val += value
        kernel.append(row)
    
    # Normalize kernel
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i][j] /= sum_val
    
    return kernel

def manual_grayscale_conversion(image):
    """Convert an image to grayscale manually."""
    height, width, _ = image.shape
    grayscale_image = np.array([[int(0.299 * image[i, j, 2] + 0.587 * image[i, j, 1] + 0.114 * image[i, j, 0]) 
                                 for j in range(width)] for i in range(height)], dtype=np.uint8)
    return grayscale_image

def apply_manual_gaussian_blur(input_path, output_path, kernel_size=3, iterations=3):
    # Load the grayscale image manually
    image = manual_grayscale_conversion(cv2.imread(input_path))
    if image is None:
        print("Error: Image not found.")
        return
    
    # Generate the Gaussian kernel
    kernel = generate_gaussian_kernel(kernel_size)
    
    # Apply Gaussian Blur iteratively
    for _ in range(iterations):
        height, width = len(image), len(image[0])
        new_image = [[0] * width for _ in range(height)]
        offset = kernel_size // 2
        
        for i in range(offset, height - offset):
            for j in range(offset, width - offset):
                sum_val = 0
                for ki in range(kernel_size):
                    for kj in range(kernel_size):
                        sum_val += image[i + ki - offset][j + kj - offset] * kernel[ki][kj]
                new_image[i][j] = int(sum_val)
        
        image = new_image
    
    # Convert back to format suitable for saving
    image = np.array([[max(0, min(255, pixel)) for pixel in row] for row in image], dtype=np.uint8)
    cv2.imwrite(output_path, image)
    print(f"Gaussian blurred image saved at: {output_path}")

def manual_resize(image_path, output_path, new_width=600, new_height=1200):
    # Load RGB image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    # Get original dimensions
    old_height, old_width, _ = image.shape

    # Create empty resized image
    resized = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # Compute scaling factors
    x_scale = old_width / new_width
    y_scale = old_height / new_height

    # Nearest-neighbor interpolation (Manual Implementation)
    for y in range(new_height):
        for x in range(new_width):
            src_x = int(x * x_scale)
            src_y = int(y * y_scale)
            resized[y, x] = image[src_y, src_x]

    # Save resized image
    cv2.imwrite(output_path, resized)
    print(f"Resized image saved at: {output_path}")

def manual_histogram_equalization(image_path, output_path):
    # Load resized image
    image = manual_grayscale_conversion(cv2.imread(image_path))
    if image is None:
        print("Error: Image not found.")
        return

    # Compute histogram
    hist = np.zeros(256, dtype=int)
    for pixel in image.flatten():
        hist[pixel] += 1

    # Compute cumulative distribution function (CDF)
    cdf = np.cumsum(hist)
    cdf_min = np.min(cdf[cdf > 0])  # Avoid division by zero

    # Normalize CDF to scale pixel values
    equalized_image = np.zeros_like(image)
    total_pixels = image.size

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            equalized_image[y, x] = ((cdf[image[y, x]] - cdf_min) / (total_pixels - cdf_min)) * 255

    equalized_image = equalized_image.astype(np.uint8)

    # Save the contrast-equalized image
    cv2.imwrite(output_path, equalized_image)
    print(f"Contrast-equalized image saved at: {output_path}")

def normalize_brightness(image):
    """Normalize brightness by scaling pixel values to a fixed range."""
    mean_desired = 128  # Set a common mean brightness
    std_desired = 50    # Set a common contrast level

    # Compute mean and standard deviation
    mean, std = np.mean(image), np.std(image)

    # Normalize image intensity to the desired mean and std
    normalized_image = ((image - mean) / (std + 1e-8)) * std_desired + mean_desired

    # Clip values to be in the valid range [0, 255]
    normalized_image = np.clip(normalized_image, 0, 255).astype(np.uint8)

    return normalized_image

def manual_brightness_normalization(image_path, output_path):
    # Load the grayscale image
    image = manual_grayscale_conversion(cv2.imread(image_path))
    if image is None:
        print("Error: Image not found.")
        return

    # Normalize brightness
    normalized_image = normalize_brightness(image)

    # Save the brightness-normalized image
    cv2.imwrite(output_path, normalized_image)
    print(f"Brightness-normalized image saved at: {output_path}")

def manual_median_blur(image_path, output_path, kernel_size=9):
    # Load contrast-enhanced image
    image = manual_grayscale_conversion(cv2.imread(image_path))
    if image is None:
        print("Error: Image not found.")
        return

    # Pad the image to handle borders
    pad = kernel_size // 2
    padded_image = np.pad(image, pad, mode='edge')

    # Create output image
    blurred_image = np.zeros_like(image)

    # Apply median filter manually
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # Extract neighborhood pixels
            neighborhood = padded_image[y:y + kernel_size, x:x + kernel_size].flatten()
            # Compute median and assign to output image
            blurred_image[y, x] = np.median(neighborhood)

    # Save the blurred image
    cv2.imwrite(output_path, blurred_image)
    print(f"Blurred image saved at: {output_path}")

def manual_threshold(image_path, output_path, threshold_value=197):
    # Load the blurred image
    image = manual_grayscale_conversion(cv2.imread(image_path))
    if image is None:
        print("Error: Image not found.")
        return

    # Create an empty binary image
    thresholded_image = np.zeros_like(image, dtype=np.uint8)

    # Apply thresholding manually
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y, x] > threshold_value:
                thresholded_image[y, x] = 255  # Set to white (255) if above threshold
            else:
                thresholded_image[y, x] = 0  # Set to black (0) otherwise

    # Save the thresholded image
    cv2.imwrite(output_path, thresholded_image)
    print(f"Thresholded image saved at: {output_path}")

def remove_small_noise(image_path, output_path, min_area_factor=0.001):
    image = manual_grayscale_conversion(cv2.imread(image_path))
    if image is None:
        print("Error: Image not found.")
        return
    
    height, width = len(image), len(image[0])
    min_area = max(height, width) * min_area_factor  # Adaptive area threshold
    labels = np.zeros((height, width), dtype=int)
    label = 1
    components = {}
    
    def flood_fill(x, y, label):
        stack = [(x, y)]
        pixels = []
        min_x, min_y, max_x, max_y = x, y, x, y
        
        while stack:
            cx, cy = stack.pop()
            if labels[cx, cy] == 0 and image[cx, cy] == 255:
                labels[cx, cy] = label
                pixels.append((cx, cy))
                min_x, min_y = min(min_x, cx), min(min_y, cy)
                max_x, max_y = max(max_x, cx), max(max_y, cy)
                
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < height and 0 <= ny < width:
                        stack.append((nx, ny))
        
        return pixels, (min_x, min_y, max_x, max_y)
    
    for i in range(height):
        for j in range(width):
            if image[i][j] == 255 and labels[i][j] == 0:
                pixels, bbox = flood_fill(i, j, label)
                components[label] = (pixels, bbox)
                label += 1
    
    cleaned_image = np.zeros_like(image, dtype=np.uint8)
    for label, (pixels, bbox) in components.items():
        area = len(pixels)
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        
        if area >= min_area or (bbox_width > 30 and bbox_height > 30):
            for x, y in pixels:
                cleaned_image[x, y] = 255
    
    cv2.imwrite(output_path, cleaned_image)
    print(f"Cleaned image saved at: {output_path}")

def manual_erosion(image, kernel):
    pad_h, pad_w = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    eroded_image = np.zeros_like(image)
    for y in range(pad_h, padded_image.shape[0] - pad_h):
        for x in range(pad_w, padded_image.shape[1] - pad_w):
            region = padded_image[y - pad_h:y + pad_h + 1, x - pad_w:x + pad_w + 1]
            if np.all(region[kernel == 1] == 255):  # Apply kernel condition
                eroded_image[y - pad_h, x - pad_w] = 255
            else:
                eroded_image[y - pad_h, x - pad_w] = 0
    return eroded_image

def manual_dilation(image, kernel):
    pad_h, pad_w = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    dilated_image = np.zeros_like(image)
    for y in range(pad_h, padded_image.shape[0] - pad_h):
        for x in range(pad_w, padded_image.shape[1] - pad_w):
            region = padded_image[y - pad_h:y + pad_h + 1, x - pad_w:x + pad_w + 1]
            if np.any(region[kernel == 1] == 255):  # Apply kernel condition
                dilated_image[y - pad_h, x - pad_w] = 255
            else:
                dilated_image[y - pad_h, x - pad_w] = 0
    return dilated_image

def manual_opening(threshold_output_path, opened_output_path, kernel_size=3):
    image = manual_grayscale_conversion(cv2.imread(threshold_output_path))
    if image is None:
        print("Error: Image not found.")
        return

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply manual erosion
    eroded_image = manual_erosion(image, kernel)

    # Apply manual dilation
    opened_image = manual_dilation(eroded_image, kernel)

    cv2.imwrite(opened_output_path, opened_image)
    print(f"Opened image saved at: {opened_output_path}")

def compute_sobel(image_path, sobel_x_path, sobel_y_path):
    # Load the thresholded image
    image = manual_grayscale_conversion(cv2.imread(image_path))
    if image is None:
        print("Error: Image not found.")
        return

    # Sobel kernels
    sobel_x = np.array([[-1,  0,  1],
                         [-2,  0,  2],
                         [-1,  0,  1]])

    sobel_y = np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]])

    # Pad the image
    pad = 1
    padded_image = np.pad(image, pad, mode='edge')

    # Initialize gradient images
    grad_x = np.zeros_like(image, dtype=np.float32)
    grad_y = np.zeros_like(image, dtype=np.float32)

    # Apply Sobel filter manually
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            region = padded_image[y:y+3, x:x+3]
            grad_x[y, x] = np.sum(region * sobel_x)
            grad_y[y, x] = np.sum(region * sobel_y)

    # Normalize and convert to uint8
    grad_x = np.abs(grad_x)
    grad_y = np.abs(grad_y)

    grad_x = (grad_x / np.max(grad_x) * 255).astype(np.uint8)
    grad_y = (grad_y / np.max(grad_y) * 255).astype(np.uint8)

    # Save Sobel gradient images
    cv2.imwrite(sobel_x_path, grad_x)
    cv2.imwrite(sobel_y_path, grad_y)
    print(f"Sobel gradients saved at: {sobel_x_path}, {sobel_y_path}")

def compute_gradient_magnitude_direction(sobel_x_path, sobel_y_path, magnitude_path, direction_path):
    # Load Sobel images
    sobel_x_path = cv2.imread(sobel_x_path)
    sobel_y_path = cv2.imread(sobel_y_path)
    grad_x = manual_grayscale_conversion(sobel_x_path)
    grad_y = manual_grayscale_conversion(sobel_y_path)

    # Convert to float32 NumPy arrays
    grad_x = np.array(grad_x, dtype=np.float32)
    grad_y = np.array(grad_y, dtype=np.float32)

    # Compute magnitude and direction
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x) * (180 / np.pi)  # Convert to degrees

    # Normalize magnitude
    magnitude = (magnitude / np.max(magnitude) * 255).astype(np.uint8)

    # Save results
    cv2.imwrite(magnitude_path, magnitude)
    np.save(direction_path, direction)  # Save direction as .npy file
    print(f"Gradient magnitude & direction saved at: {magnitude_path}, {direction_path}")

def non_maximum_suppression(magnitude_path, direction_path, nms_path):
    magnitude_path = cv2.imread(magnitude_path)
    # Load magnitude & direction
    magnitude = manual_grayscale_conversion(magnitude_path)
    magnitude = np.array(magnitude, dtype=np.float32)
    direction = np.load(direction_path)

    # Pad image
    padded_mag = np.pad(magnitude, 1, mode='constant')

    # Create NMS output
    nms_image = np.zeros_like(magnitude, dtype=np.uint8)

    # Iterate through pixels
    for y in range(1, magnitude.shape[0] + 1):
        for x in range(1, magnitude.shape[1] + 1):
            angle = direction[y-1, x-1] % 180  # Normalize angle
            q = 255
            r = 255

            # Define neighbors based on gradient direction
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q = padded_mag[y, x+1]
                r = padded_mag[y, x-1]
            elif 22.5 <= angle < 67.5:
                q = padded_mag[y-1, x+1]
                r = padded_mag[y+1, x-1]
            elif 67.5 <= angle < 112.5:
                q = padded_mag[y-1, x]
                r = padded_mag[y+1, x]
            elif 112.5 <= angle < 157.5:
                q = padded_mag[y-1, x-1]
                r = padded_mag[y+1, x+1]

            # Suppress non-maximum values
            if magnitude[y-1, x-1] >= q and magnitude[y-1, x-1] >= r:
                nms_image[y-1, x-1] = magnitude[y-1, x-1]

    # Save result
    cv2.imwrite(nms_path, nms_image)
    print(f"NMS image saved at: {nms_path}")

def double_threshold(nms_path, strong_path, weak_path, low_thresh=50, high_thresh=100):
    # Load NMS image
    nms = manual_grayscale_conversion(cv2.imread(nms_path))

    # Create thresholded images
    strong_edges = (nms >= high_thresh).astype(np.uint8) * 255
    weak_edges = ((nms >= low_thresh) & (nms < high_thresh)).astype(np.uint8) * 255

    # Save results
    cv2.imwrite(strong_path, strong_edges)
    cv2.imwrite(weak_path, weak_edges)
    print(f"Double threshold images saved at: {strong_path}, {weak_path}")

def edge_tracking_by_hysteresis(strong_path, weak_path, final_edge_path):
    # Load strong and weak edge maps
    strong_edges = manual_grayscale_conversion(cv2.imread(strong_path))
    weak_edges = manual_grayscale_conversion(cv2.imread(weak_path))

    # Copy strong edges into final output
    final_edges = strong_edges.copy()

    # Get image dimensions
    height, width = weak_edges.shape

    # Define 8-connected neighborhood offsets
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    # Iterate through the weak edge pixels
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if weak_edges[y, x] == 255:  # If pixel is a weak edge
                # Check if it is connected to a strong edge
                for dy, dx in neighbors:
                    if strong_edges[y + dy, x + dx] == 255:
                        final_edges[y, x] = 255  # Convert to strong edge
                        break
                else:
                    final_edges[y, x] = 0  # Remove if not connected to a strong edge

    # Save the final edge-detected image
    cv2.imwrite(final_edge_path, final_edges)
    print(f"Final edge-detected image saved at: {final_edge_path}")

def get_contours_manually(edge_image):
    height, width = edge_image.shape
    visited = np.zeros_like(edge_image, dtype=bool)
    contours = []
    min_contour_length = max(width, height) * 0.01  # Adaptive threshold
    
    def trace_contour(x, y):
        contour = []
        stack = [(x, y)]
        
        while stack:
            cx, cy = stack.pop()
            if visited[cy, cx]:
                continue
            visited[cy, cx] = True
            contour.append([cx, cy])
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < width and 0 <= ny < height and edge_image[ny, nx] > 0 and not visited[ny, nx]:
                        stack.append((nx, ny))
        
        return np.array(contour, dtype=np.int32)
    
    for y in range(height):
        for x in range(width):
            if edge_image[y, x] > 0 and not visited[y, x]:
                contour = trace_contour(x, y)
                if len(contour) > min_contour_length:
                    contours.append(contour)
    
    return contours

def connect_disjoint_segments(contours, min_dist_factor=0.05):
    merged_contours = []
    used = set()
    
    if not contours:
        return []
    
    width, height = max(max(c[:, 0]) for c in contours), max(max(c[:, 1]) for c in contours)
    min_dist = max(width, height) * min_dist_factor  # Adaptive distance threshold
    
    for i in range(len(contours)):
        if i in used:
            continue
        cnt1 = contours[i]
        merged_contour = cnt1.tolist()
        
        for j in range(i + 1, len(contours)):
            if j in used:
                continue
            cnt2 = contours[j]
            x1, y1 = merged_contour[-1]
            x2, y2 = cnt2[0]
            dist = np.hypot(x2 - x1, y2 - y1)
            if dist < min_dist:
                merged_contour.extend(cnt2.tolist())
                used.add(j)
        
        merged_contours.append(np.array(merged_contour, dtype=np.int32))
        used.add(i)
    
    return merged_contours

def enhance_edges(final_edge_path, output_edge_path, min_dist_factor=0.05):
    edge_image = manual_grayscale_conversion(cv2.imread(final_edge_path))
    if edge_image is None:
        print("Error: Edge image not found.")
        return
    
    contours = get_contours_manually(edge_image)
    merged_contours = connect_disjoint_segments(contours, min_dist_factor)
    
    enhanced_edges = np.zeros_like(edge_image, dtype=np.uint8)
    for contour in merged_contours:
        for i in range(len(contour) - 1):
            x1, y1 = contour[i]
            x2, y2 = contour[i + 1]
            cv2.line(enhanced_edges, (x1, y1), (x2, y2), 255, 1)
    
    cv2.imwrite(output_edge_path, enhanced_edges)
    print(f"Enhanced edge image saved at: {output_edge_path}")

def hough_transform(edge_image_path, overlay_path, output_image_path, rho_res=1, theta_res=np.pi/360, threshold=25):
    # Load the edge-detected image
    edges = manual_grayscale_conversion(cv2.imread(edge_image_path))
    if edges is None:
        print("Error: Edge image not found.")
        return

    height, width = edges.shape
    cropped_edges = edges[int(height / 9):, :]
    cropped_height = cropped_edges.shape[0]

    max_rho = int(np.hypot(cropped_height, width))
    rhos = np.arange(-max_rho, max_rho, rho_res)
    thetas = np.arange(0, np.pi, theta_res)

    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)

    for y in range(cropped_height):
        for x in range(width):
            if cropped_edges[y, x] == 255:
                for theta_idx in range(len(thetas)):
                    theta = thetas[theta_idx]
                    rho = int(x * np.cos(theta) + (y + int(height / 8)) * np.sin(theta))
                    rho_idx = np.argmin(np.abs(rhos - rho))
                    accumulator[rho_idx, theta_idx] += 1

    detected_lines = []
    for rho_idx in range(accumulator.shape[0]):
        for theta_idx in range(accumulator.shape[1]):
            if accumulator[rho_idx, theta_idx] >= threshold:
                rho = rhos[rho_idx]
                theta = thetas[theta_idx]
                detected_lines.append((rho, theta, accumulator[rho_idx, theta_idx]))

    detected_lines = sorted(detected_lines, key=lambda x: x[2], reverse=True)[:8]
    detected_lines = [(rho, theta) for rho, theta, _ in detected_lines]

    def cluster_lines(lines, rho_threshold=30, theta_threshold=np.pi / 180 * 10):
        clusters = []
        for rho, theta in lines:
            added = False
            for cluster in clusters:
                avg_rho, avg_theta = np.mean(cluster, axis=0)
                if abs(avg_rho - rho) < rho_threshold and abs(avg_theta - theta) < theta_threshold:
                    cluster.append((rho, theta))
                    added = True
                    break
            if not added:
                clusters.append([(rho, theta)])
        return [tuple(np.mean(cluster, axis=0)) for cluster in clusters]

    lane_lines = cluster_lines(detected_lines)

    overlay_image = cv2.imread(overlay_path)

    for rho, theta in lane_lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(overlay_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite(output_image_path, overlay_image)
    print(f"Overlayed lane lines saved at: {output_image_path}")

    return lane_lines

def find_line_intersections(lines):
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            rho1, theta1 = lines[i]
            rho2, theta2 = lines[j]
            
            det = np.cos(theta1) * np.sin(theta2) - np.cos(theta2) * np.sin(theta1)
            if abs(det) < 1e-10:
                continue
            
            x = (rho2 * np.sin(theta1) - rho1 * np.sin(theta2)) / det
            y = (rho1 * np.cos(theta2) - rho2 * np.cos(theta1)) / det
            intersections.append((x, y))
    return intersections

def compute_centroid(points):
    if not points:
        return (0, 0)
    x_sum, y_sum = sum(x for x, y in points), sum(y for x, y in points)
    return (x_sum / len(points), y_sum / len(points))

def compute_fit_quality(lines):
    intersections = find_line_intersections(lines)
    centroid = compute_centroid(intersections)
    
    total_distance = sum(((x - centroid[0])**2 + (y - centroid[1])**2) ** 0.5 for x, y in intersections)
    return total_distance

def bgr_to_rgb(image):
    """Manually converts a BGR image to RGB."""
    return image[:, :, ::-1]  # Reverse the last axis

def visualize_steps(original_image, edges, contour_image, final_lane_image):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    axes[0, 0].imshow(bgr_to_rgb(original_image))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(edges, cmap='gray')
    axes[0, 1].set_title('Detected Edges')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(bgr_to_rgb(contour_image))
    axes[1, 0].set_title('Filtered Contours')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(bgr_to_rgb(final_lane_image))
    axes[1, 1].set_title('Final Quadratic Lane Curves')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

def process_all_images(input_dir, output_dir, task, output_csv):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    csv_file = os.path.join(os.getcwd(), output_csv)  # Task2.csv in the main working directory
    
    # Create the CSV file at the beginning and write the header
    if task == 2 and not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Image_Name", "Line_Fit_Score"])
    
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            hough_output = os.path.join(output_dir, filename)  # Keep the same filename
            
            print(f"Processing {filename}...")
            
            temp_files = [
                "temp_L.png", "temp_blur.png", "temp_resize.png", "temp_hist_eq.png", "temp_brightness.png",
                "temp_median_blur.png", "temp_threshold.png", "temp_noise_removed.png", "temp_opening.png",
                "temp_sobel_x.png", "temp_sobel_y.png", "temp_magnitude.png", "temp_direction.npy", "temp_nms.png",
                "temp_strong.png", "temp_weak.png", "temp_final_edge.png", "temp_enhanced.png"
            ]
            
            extract_L_channel(input_path, temp_files[0])
            apply_manual_gaussian_blur(temp_files[0], temp_files[1])
            manual_resize(temp_files[1], temp_files[2])
            manual_histogram_equalization(temp_files[2], temp_files[3])
            manual_brightness_normalization(temp_files[3], temp_files[4])
            manual_median_blur(temp_files[4], temp_files[5])
            manual_threshold(temp_files[5], temp_files[6])
            remove_small_noise(temp_files[6], temp_files[7])
            manual_opening(temp_files[7], temp_files[8])
            compute_sobel(temp_files[8], temp_files[9], temp_files[10])
            compute_gradient_magnitude_direction(temp_files[9], temp_files[10], temp_files[11], temp_files[12])
            non_maximum_suppression(temp_files[11], temp_files[12], temp_files[13])
            double_threshold(temp_files[13], temp_files[14], temp_files[15])
            edge_tracking_by_hysteresis(temp_files[14], temp_files[15], temp_files[16])
            enhance_edges(temp_files[16], temp_files[17])
            
            manual_resize(input_path, input_path)
            detected_lines = hough_transform(temp_files[17], input_path, hough_output)
            
            print(f"Saved output for {filename} at {hough_output}")
            
            if task == 2:
                fit_quality = compute_fit_quality(detected_lines)
                print(f"Line Fit Quality Score for {filename}: {fit_quality}")
                
                # Append the data to CSV file immediately
                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([filename, fit_quality])
                
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 grassy_straight.py <task> <input_img_dir> <output_img_dir/output_csv>")
        sys.exit(1)
    
    task = int(sys.argv[1])
    input_img_dir = sys.argv[2]
    output_param = sys.argv[3]
    
    if task in [1, 2]:
        process_all_images(input_img_dir, output_param if task == 1 else os.getcwd(), task, "Task2.csv")
    else:
        print("Invalid task number. Use 1 for lane detection, 2 for line fit scoring.")
