import cv2
import numpy as np
import matplotlib.pyplot as plt

def manual_grayscale_conversion(image):
    """Convert an image to grayscale manually."""
    height, width, _ = image.shape
    grayscale_image = np.array([[int(0.299 * image[i, j, 2] + 0.587 * image[i, j, 1] + 0.114 * image[i, j, 0]) 
                                 for j in range(width)] for i in range(height)], dtype=np.uint8)
    return grayscale_image

def rgb_to_lab_manual(image):
    # Normalize RGB values to [0, 1]
    image = image / 255.0
    
    # Convert RGB to XYZ
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    
    shape = image.shape
    image_reshaped = image.reshape(-1, 3)
    xyz = np.dot(image_reshaped, M.T).reshape(shape)
    
    # Normalize for D65 white point
    X, Y, Z = xyz[:, :, 0] / 0.95047, xyz[:, :, 1], xyz[:, :, 2] / 1.08883
    
    # Apply the standard LAB conversion
    def f(t):
        delta = 6/29
        return np.where(t > delta**3, np.cbrt(t), (t / (3 * delta**2)) + (4/29))
    
    L = (116 * f(Y)) - 16
    A = 500 * (f(X) - f(Y))
    B = 200 * (f(Y) - f(Z))
    
    return L, A, B

def extract_a_channel_manual(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return
    
    # Convert image to LAB manually
    L, A, B = rgb_to_lab_manual(image.astype(np.float32))
    
    # Normalize A-channel to 8-bit
    A = ((A - np.min(A)) / (np.max(A) - np.min(A)) * 255).astype(np.uint8)
    
    cv2.imwrite(output_path, A)
    print(f"A-channel extracted and saved at: {output_path}")

def manual_resize(image_path, output_path, new_width=600, new_height=1200):
    # Load the grayscale A-channel image
    image = manual_grayscale_conversion(cv2.imread(image_path))
    if image is None:
        print("Error: Image not found.")
        return

    # Get original dimensions
    old_height, old_width = image.shape

    # Create empty resized image
    resized_image = np.zeros((new_height, new_width), dtype=np.uint8)

    # Compute scaling factors
    x_scale = old_width / new_width
    y_scale = old_height / new_height

    # Nearest-neighbor interpolation (Manual Implementation)
    for y in range(new_height):
        for x in range(new_width):
            src_x = int(x * x_scale)
            src_y = int(y * y_scale)
            resized_image[y, x] = image[src_y, src_x]

    # Save resized image
    cv2.imwrite(output_path, resized_image)
    print(f"Resized image saved at: {output_path}")

def manual_gaussian_kernel(kernel_size, sigma):
    kernel = []
    sum_val = 0
    center = kernel_size // 2
    
    for i in range(kernel_size):
        row = []
        for j in range(kernel_size):
            x, y = i - center, j - center
            value = (1 / (2 * 3.141592653589793 * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
            row.append(value)
            sum_val += value
        kernel.append(row)
    
    # Normalize kernel
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i][j] /= sum_val
    
    return kernel

def manual_gaussian_blur(image_path, output_path, kernel_size=3, sigma=1.0, iterations=2):
    image = manual_grayscale_conversion(cv2.imread(image_path))
    if image is None:
        print("Error: Image not found.")
        return
    
    kernel = manual_gaussian_kernel(kernel_size, sigma)
    pad = kernel_size // 2
    
    for _ in range(iterations):
        padded_image = np.pad(image, pad, mode='edge')
        blurred_image = np.zeros_like(image)
        
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                sum_val = 0
                for i in range(kernel_size):
                    for j in range(kernel_size):
                        sum_val += padded_image[y + i, x + j] * kernel[i][j]
                blurred_image[y, x] = int(sum_val)
        
        image = blurred_image.astype(np.uint8)
    
    cv2.imwrite(output_path, image)
    print(f"Gaussian blurred image saved at: {output_path}")

def otsu_threshold(image_path, output_path):
    # Load the Gaussian-blurred image
    image = manual_grayscale_conversion(cv2.imread(image_path))
    if image is None:
        print("Error: Image not found.")
        return

    # Compute histogram
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0,256])

    # Compute Otsu’s threshold
    total_pixels = image.size
    sum_total = np.sum(np.arange(256) * hist)
    sum_background, weight_background, weight_foreground = 0, 0, 0
    max_variance, otsu_threshold_value = 0, 0

    for t in range(256):
        weight_background += hist[t]
        if weight_background == 0:
            continue

        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break

        sum_background += t * hist[t]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        # Compute between-class variance
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

        if variance > max_variance:
            max_variance = variance
            otsu_threshold_value = t

    # Apply Otsu’s thresholding
    thresholded_image = np.where(image > otsu_threshold_value, 255, 0).astype(np.uint8)

    # Save the thresholded image
    cv2.imwrite(output_path, thresholded_image)
    print(f"Otsu’s thresholded image saved at: {output_path}")
 
def remove_small_noise(image_path, output_path, min_area=100):
    """
    Remove small noise using manual connected components analysis.
    """
    image = manual_grayscale_conversion(cv2.imread(image_path))
    if image is None:
        print("Error: Image not found.")
        return
    
    height, width = len(image), len(image[0])
    labels = [[0] * width for _ in range(height)]
    label = 1
    components = {}
    
    def flood_fill(x, y, label):
        stack = [(x, y)]
        pixels = []
        while stack:
            cx, cy = stack.pop()
            if labels[cx][cy] == 0 and image[cx][cy] == 255:
                labels[cx][cy] = label
                pixels.append((cx, cy))
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < height and 0 <= ny < width:
                        stack.append((nx, ny))
        return pixels
    
    # Label connected components
    for i in range(height):
        for j in range(width):
            if image[i][j] == 255 and labels[i][j] == 0:
                pixels = flood_fill(i, j, label)
                components[label] = pixels
                label += 1
    
    # Create cleaned image
    cleaned_image = [[0] * width for _ in range(height)]
    for label, pixels in components.items():
        if len(pixels) >= min_area:
            for x, y in pixels:
                cleaned_image[x][y] = 255
    
    # Convert to NumPy array and save
    cleaned_image = np.array(cleaned_image, dtype=np.uint8)
    cv2.imwrite(output_path, cleaned_image)
    print(f"Cleaned image saved at: {output_path}")

def compute_sobel(image_path, sobel_x_path, sobel_y_path, magnitude_path, direction_path):
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

    # Compute magnitude and direction
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x) * (180 / np.pi)  # Convert to degrees

    # Normalize magnitude
    magnitude = (magnitude / np.max(magnitude) * 255).astype(np.uint8)

    # Save results
    cv2.imwrite(sobel_x_path, grad_x.astype(np.uint8))
    cv2.imwrite(sobel_y_path, grad_y.astype(np.uint8))
    cv2.imwrite(magnitude_path, magnitude)
    np.save(direction_path, direction)  # Save direction as .npy file
    print(f"Sobel edge gradients saved at: {magnitude_path} and {direction_path}")

def non_maximum_suppression(magnitude_path, direction_path, nms_path):
    # Load magnitude & direction
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

def manual_erode(image, kernel):
    pad = len(kernel) // 2
    padded_image = np.pad(image, pad, mode='edge')
    eroded_image = np.zeros_like(image)
    
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            region = padded_image[y:y+len(kernel), x:x+len(kernel)]
            eroded_image[y, x] = np.min(region)
    
    return eroded_image

def manual_dilate(image, kernel):
    pad = len(kernel) // 2
    padded_image = np.pad(image, pad, mode='edge')
    dilated_image = np.zeros_like(image)
    
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            region = padded_image[y:y+len(kernel), x:x+len(kernel)]
            dilated_image[y, x] = np.max(region)
    
    return dilated_image

def simple_morphological_opening(edge_image_path, output_path, kernel_size=1):
    # Load the edge-detected image
    edges = manual_grayscale_conversion(cv2.imread(edge_image_path))
    if edges is None:
        print("Error: Edge image not found.")
        return
    
    # Define kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Apply Opening (Erosion → Dilation) manually
    opened = manual_erode(edges, kernel)
    opened = manual_dilate(opened, kernel)
    
    # Save cleaned edge map
    cv2.imwrite(output_path, opened)
    print(f"Simple opened image saved at: {output_path}")

def hough_transform(edge_image_path, output_image_path, rho_res=1, theta_res=np.pi/360, threshold=50):
    # Load the edge-detected image
    edges = manual_grayscale_conversion(cv2.imread(edge_image_path))
    if edges is None:
        print("Error: Edge image not found.")
        return

    height, width = edges.shape

    # **Crop the image to process only the bottom 7/8th**
    cropped_edges = edges[int(height / 9):, :]  # Keep only the lower 7/8th
    cropped_height = cropped_edges.shape[0]

    # Define rho and theta ranges
    max_rho = int(np.hypot(cropped_height, width))
    rhos = np.arange(-max_rho, max_rho, rho_res)
    thetas = np.arange(0, np.pi, theta_res)

    # Create accumulator array
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)

    # Manual Hough Transform: Voting in the accumulator
    for y in range(cropped_height):
        for x in range(width):
            if cropped_edges[y, x] == 255:  # Edge pixel detected
                for theta_idx in range(len(thetas)):
                    theta = thetas[theta_idx]
                    rho = int(x * np.cos(theta) + (y + int(height / 8)) * np.sin(theta))  # Adjust y-coordinates
                    rho_idx = np.argmin(np.abs(rhos - rho))
                    accumulator[rho_idx, theta_idx] += 1

    # Find peaks in accumulator
    detected_lines = []
    for rho_idx in range(accumulator.shape[0]):
        for theta_idx in range(accumulator.shape[1]):
            if accumulator[rho_idx, theta_idx] >= threshold:
                rho = rhos[rho_idx]
                theta = thetas[theta_idx]
                detected_lines.append((rho, theta))

    # **Step 1: Cluster similar lines to avoid multiple detections per lane**
    def cluster_lines(lines, rho_threshold=30, theta_threshold=np.pi / 180 * 10):
        """Clusters similar lines based on rho and theta values to get a single lane line."""
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

        # Compute the average line for each cluster
        filtered_lines = [tuple(np.mean(cluster, axis=0)) for cluster in clusters]
        return filtered_lines

    filtered_lines = cluster_lines(detected_lines)

    # **Step 2: Select only lane-related lines based on slope**
    def filter_lane_lines(lines, min_slope=0.3, max_slope=3):
        """Filters lines that are nearly vertical/horizontal and keeps only valid lane lines."""
        valid_lines = []
        for rho, theta in lines:
            slope = np.tan(theta)
            if min_slope < abs(slope) < max_slope:  # Avoid horizontal and very steep lines
                valid_lines.append((rho, theta))
        return valid_lines

    lane_lines = filter_lane_lines(filtered_lines)

    # Load original image to draw lines
    original_image = cv2.imread(edge_image_path)

    # **Step 3: Draw only the selected lane lines**
    for rho, theta in lane_lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(original_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Save the result
    cv2.imwrite(output_image_path, original_image)
    print(f"Filtered lane lines detected and saved at: {output_image_path}")

def process_all_images(input_dir, output_dir, task, output_csv):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    csv_file = os.path.join(os.getcwd(), output_csv)
    
    if task == 2 and not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Image_Name", "Line_Fit_Score"])
    
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_image_path = os.path.join(output_dir, filename)
            visualization_output_path = os.path.join(output_dir, f"viz_{filename}")
            
            print(f"Processing {filename}...")
            
            temp_files = [
                "temp_a_channel.png", "temp_resize.png", "temp_blur.png", "temp_threshold.png", "temp_noise_removed.png",
                "temp_sobel_x.png", "temp_sobel_y.png", "temp_magnitude.png", "temp_direction.npy", "temp_nms.png",
                "temp_strong.png", "temp_weak.png", "temp_final_edge.png", "temp_opened.png"
            ]
            
            extract_a_channel_manual(input_path, temp_files[0])
            manual_resize(temp_files[0], temp_files[1])
            manual_gaussian_blur(temp_files[1], temp_files[2])
            otsu_threshold(temp_files[2], temp_files[3])
            remove_small_noise(temp_files[3], temp_files[4])
            compute_sobel(temp_files[4], temp_files[5], temp_files[6], temp_files[7], temp_files[8])
            non_maximum_suppression(temp_files[7], temp_files[8], temp_files[9])
            double_threshold(temp_files[9], temp_files[10], temp_files[11])
            edge_tracking_by_hysteresis(temp_files[10], temp_files[11], temp_files[12])
            simple_morphological_opening(temp_files[12], temp_files[13])
            
            hough_transform(temp_files[13], visualization_output_path)
            
            if task == 1:
                print(f"Processed image saved at: {output_image_path}")
            elif task == 2:
                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([filename, "Placeholder_Score"])
                print(f"Curve data saved for {filename} in {csv_file}")
            
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 main.py <task> <input_img_dir> <output_img_dir/output_csv>")
        sys.exit(1)
    
    task = int(sys.argv[1])
    input_img_dir = sys.argv[2]
    output_param = sys.argv[3]
    
    if task in [1, 2]:
        process_all_images(input_img_dir, output_param if task == 1 else os.getcwd(), task, "Task2.csv")
    else:
        print("Invalid task number. Use 1 for lane detection, 2 for line fit scoring.")
