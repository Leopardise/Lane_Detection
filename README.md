# Lane Boundary Detection

Lane detection using classical image processing techniques

---

## Overview

This repository contains the implementation and detailed analysis of lane boundary detection algorithms using image processing techniques.

## Importance of Lane Detection

Lane detection is crucial in autonomous driving, ensuring safe navigation, and in sports analytics for tracking player positions and analyzing movement patterns.

## Dataset

The dataset comprises images from video recordings taken at IIT Delhi's football ground. The dataset includes:

* **Brick-Lane Images:** Straight and curved lanes with reddish-brown surfaces.
* **Athletic Track Images:** Straight and curved lanes on greenish grassy fields with white chalk.

### Dataset Characteristics

| Property            | Brick-Lane Images          | Athletic Track Images                           |
| ------------------- | -------------------------- | ----------------------------------------------- |
| Surface             | Reddish-brown              | Green grassy field with white chalk lanes       |
| Environment         | Trees and bushes nearby    | Open sports field                               |
| Shadow/Illumination | High contrast with shadows | Uniformly lit                                   |
| Colour Space (LAB)  | High A-channel values      | Low A-channel values                            |
| Brightness          | Low                        | High                                            |
| Contrast            | High                       | Low                                             |
| Noise Type          | Additive noise (leaves)    | Multiplicative noise (salt-and-pepper on grass) |

## Project Structure

```
col780/
├── lane_dataset/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── ...
├── grassy/ (automatically created)
├── bricky/ (automatically created)
├── main.py
├── Task2.csv/
└── Task1_output_images.csv/
```

## Methods and Workflow

### Image Classification

Images are classified into "bricky" and "grassy" based on contrast, brightness, and LAB A-channel values:

* **Bricky:** High contrast, low brightness, high A-channel.
* **Grassy:** Low contrast, high brightness, low A-channel.

### Preprocessing Steps

#### Brick-Lane Images

* Extract A-channel (LAB)
* Resize to 600×1200 pixels
* Gaussian blur (two iterations of 3×3 kernel)
* Otsu's thresholding
* Connected Component Analysis (CCA)

#### Athletic Track Images

* Extract L-channel (LAB)
* Gaussian and Median blur
* Resize to 600×1200 pixels
* Contrast Limited Adaptive Histogram Equalization (CLAHE)
* Brightness normalization
* Fixed-value thresholding
* CCA and morphological opening

### Edge Detection

For both image types:

* Sobel Edge Detection
* Non-Maximum Suppression (NMS)
* Double Thresholding (DT)
* Edge Tracking by Hysteresis
* Morphological Opening
* Connected Component Analysis (CCA)

### Lane Boundary Detection

* **Hough Transform** for straight lanes
* **Polynomial Fit** (degree-2) for curved lanes
* **Clustering and merging** lines by distance and angle thresholds
* Angle filtering and contour detection for robustness

## Results Evaluation

Validation involved analyzing the intersections of detected lines:

* Calculated centroid of intersection points
* Computed sum of distances from intersections to centroid
* Lower sum indicates better lane detection quality

Due to limited lane detection pairs (typically 0 or 1), resulting scores mostly were zero, indicating precise detection without redundant intersections.

---

### Usage

Run the main classification and lane detection script:

```bash
python main.py
```

Ensure the `lane_dataset` folder is correctly populated with images.

---

## Author

* **Avilasha Mandal**
IIT Delhi
