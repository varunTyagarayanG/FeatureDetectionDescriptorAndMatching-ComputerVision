
# Harris Corner Detection, SIFT Feature Matching and SSD Feature Matching

This repository contains Python code that implements Harris corner detection, SIFT feature matching, and SSD (Sum of Squared Differences) feature matching for two images. The code is written using `OpenCV`, `NumPy`, and `SciPy` libraries.

## Files
- `img2.png`: The first input image.
- `img4.png`: The second input image.
- `S20220010070_FDDM_output1.png`: The first output image with Harris corners detected.
- `S20220010070_FDDM_output2.png`: The second output image with Harris corners detected.
- `S20220010070_FDDM_output3.png`: The first output image with SIFT keypoints detected.
- `S20220010070_FDDM_output4.png`: The second output image with SIFT keypoints detected.
- `S20220010070_FDDM_output5.png`: The image showing feature matching between `img2.png` and `img4.png`.

## Code Explanation

### Libraries Used
- `OpenCV`: For image processing, Harris corner detection, and SIFT feature detection.
- `NumPy`: For matrix and array operations.
- `SciPy`: For applying the Gaussian filter to smooth images.

### Steps in the Code
1. **Image Loading and Preprocessing**:
   - The images are loaded using `cv2.imread()`.
   - They are converted to grayscale using `cv2.cvtColor()`.
   - A Gaussian filter from `scipy.ndimage` is applied to smooth the images.

2. **Harris Corner Detection**:
   - The Harris corner detection algorithm is applied using `cv2.cornerHarris()`.
   - The detected corners are highlighted by dilating the corner points and marking them in the images.
   - The resulting images are saved as `S20220010070_FDDM_output1.png` and `S20220010070_FDDM_output2.png`.

3. **SIFT Feature Detection**:
   - SIFT (Scale-Invariant Feature Transform) is used to detect keypoints and compute descriptors for both smoothed images.
   - Keypoints are drawn on the images and saved as `S20220010070_FDDM_output3.png` and `S20220010070_FDDM_output4.png`.

4. **Feature Matching using SSD**:
   - A manual implementation of SSD (Sum of Squared Differences) is used to match descriptors from the two images.
   - Matches are filtered based on a ratio test (threshold = 0.75).
   - The good matches are drawn and saved as `S20220010070_FDDM_output5.png`.

## How to Run the Code
1. Ensure you have the required libraries installed:
   ```bash
   pip install opencv-python-headless numpy scipy
   ```
2. Place the input images (`img2.png` and `img4.png`) in the working directory.
3. Run the Python script.

## Output
- The script will output five images:
  - Two images with detected Harris corners.
  - Two images with detected SIFT keypoints.
  - One image showing feature matching using SSD.

## Conclusion
This code demonstrates the use of Harris corner detection, SIFT for feature detection, and a basic SSD-based feature matching technique. The results show how these techniques can be used to match features between two images effectively.
