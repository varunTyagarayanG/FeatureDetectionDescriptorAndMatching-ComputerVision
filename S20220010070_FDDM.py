import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

# Load the images
image1 = cv2.imread('./img2.png')
image2 = cv2.imread('./img4.png')

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Apply Gaussian filter (from scipy.ndimage) for smoothing
smooth1 = gaussian_filter(gray1, sigma=1)
smooth2 = gaussian_filter(gray2, sigma=1)

# Manually convert the image to float32
def convert_to_float32(image):
    rows, cols = image.shape
    float_image = np.zeros((rows, cols), dtype=np.float32)  # Create a new float32 array
    
    for i in range(rows):
        for j in range(cols):
            float_image[i, j] = float(image[i, j])  # Convert each pixel to float
            
    return float_image

# Convert the smoothed images to float32
smooth1_float = convert_to_float32(smooth1)
smooth2_float = convert_to_float32(smooth2)

# Detect Harris corners for both images
dst1 = cv2.cornerHarris(smooth1_float, blockSize=2, ksize=3, k=0.04)
dst2 = cv2.cornerHarris(smooth2_float, blockSize=2, ksize=3, k=0.04)

# Dilate the corner image to enhance corner points
dst1 = cv2.dilate(dst1, None)
dst2 = cv2.dilate(dst2, None)

# Marking the corners in white
image1[dst1 > 0.01 * dst1.max()] = [255, 140, 0] # White color
image2[dst2 > 0.01 * dst2.max()] = [255, 140, 0]  # White color

# Save the output images
cv2.imwrite('S20220010070_FDDM_output1.png', image1)
cv2.imwrite('S20220010070_FDDM_output2.png', image2)

print("Output images saved as S20220010070_FDDM_output1.png and S20220010070_FDDM_output2.png")


sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for the smoothed images
keypoints1, descriptors1 = sift.detectAndCompute(smooth1, None)
keypoints2, descriptors2 = sift.detectAndCompute(smooth2, None)

# Draw keypoints on the images
image1_with_keypoints = cv2.drawKeypoints(image1, keypoints1, None, color=(255, 105, 180))  # Pink color
image2_with_keypoints = cv2.drawKeypoints(image2, keypoints2, None, color=(255, 105, 180))  # Pink color

# Save the output images with keypoints for the second question
cv2.imwrite('S20220010070_FDDM_output3.png', image1_with_keypoints)
cv2.imwrite('S20220010070_FDDM_output4.png', image2_with_keypoints)

# Print the number of keypoints detected and descriptors
print(f"Image 1: {len(keypoints1)} keypoints detected and {descriptors1.shape[0]} descriptors computed.")
print(f"Image 2: {len(keypoints2)} keypoints detected and {descriptors2.shape[0]} descriptors computed.")

def match_features(descriptors1, descriptors2):
    matches = []
    
    for i in range(len(descriptors1)):
        distances = np.sum((descriptors2 - descriptors1[i]) ** 2, axis=1)  # SSD calculation
        sorted_indices = np.argsort(distances)  # Get indices of sorted distances
        
        if len(sorted_indices) >= 2:  # Ensure we have at least two matches
            best_idx = sorted_indices[0]
            second_best_idx = sorted_indices[1]
            best_distance = distances[best_idx]
            second_best_distance = distances[second_best_idx]
            
            if second_best_distance > 0:  # Prevent division by zero
                ratio_distance = best_distance / second_best_distance
                matches.append((i, best_idx, best_distance, second_best_distance, ratio_distance))
    
    return matches

# Get matches
matches = match_features(descriptors1, descriptors2)

# Filter good matches based on ratio distance threshold (commonly 0.75)
good_matches = [m for m in matches if m[4] < 0.75]  # Keep matches with ratio distance less than 0.75

# Draw good matches
matching_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, 
                                  [cv2.DMatch(m[0], m[1], 0) for m in good_matches], 
                                  None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

# Save the output image showing good matches
cv2.imwrite('S20220010070_FDDM_output5.png', matching_image)

# Print the number of good matches
print(f"Number of good matches: {len(good_matches)}")