import cv2
import numpy as np

# The Kuwahara filter :
# The Kuwahara filter is a digital image filter that is used for smoothing an image, there will be sharper edges after filtering the image
# It operates on the principle of tiling of images with squares and the estimation of mean and variance of pixel density intensity
# The pixel with the smallest variation is taken for the output pixel which reduces the smoothness level while preserving differences.

# Read the input image
image = cv2.imread('Q3_1_Input.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Save the grayscale image
cv2.imwrite('Q3_2_gray_image.jpg', gray_image)
print("Q3_gray_image.jpg Saved")

class KuwaharaFilter:
    def __init__(self, image, k=7):
        self.k = k
        self.image = image
        self.rows, self.columns = image.shape
        # Create a border around the image to handle edge cases
        self.border_image = cv2.copyMakeBorder(image, k // 2, k // 2, k // 2, k // 2, cv2.BORDER_REFLECT)
        self.kuwahara_image = np.zeros_like(image)

    def _get_regions(self, row, col):
        r1 = self.border_image[row:row+self.k, col:col+self.k]  # Top-left region
        r2 = self.border_image[row:row+self.k, col+self.k:col+2*self.k] # Top-right region
        r3 = self.border_image[row+self.k:row+2*self.k, col:col+self.k] # Bottom-left region
        r4 = self.border_image[row+self.k:row+2*self.k, col+self.k:col+2*self.k] # Bottom-right region
        return [r1, r2, r3, r4]

    # Calculate the mean and variance for each of the regions
    def _calculate_means_variances(self, regions):
        means = []
        variances = []

        for region in regions:
            if region.size > 0:
                means.append(np.mean(region))
                variances.append(np.var(region))
            else:
                means.append(0)
                variances.append(0)
        return means, variances

    # Applying the Kuwahara filter to each pixel in the image
    def apply(self):
        for row in range(self.rows):
            for col in range(self.columns):
                regions = self._get_regions(row, col)
                means, variances = self._calculate_means_variances(regions)

                min_variance_index = np.argmin(variances)
                self.kuwahara_image[row, col] = means[min_variance_index]

        self.kuwahara_image = np.clip(self.kuwahara_image, 0, 255).astype(np.uint8)
        return self.kuwahara_image

print("Kuwahara Filter ........")
kuwahara_image = KuwaharaFilter(gray_image.copy(), k=7).apply()
cv2.imwrite('Q3_2_kuwahara_image.jpg', kuwahara_image)
print("Q3_kuwahara_image.jpg Saved")
