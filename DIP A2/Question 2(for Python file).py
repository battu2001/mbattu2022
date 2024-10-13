import cv2
import numpy as np

# Read the input image
image = cv2.imread('Q2_1_Input.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Save the grayscale image
cv2.imwrite('Q2_2_gray_image.jpg', gray_image)

print("Q2_gray_image.jpg Saved")

class Dithering:
    def __init__(self, image):
        self.image = image
        self.rows, self.columns = image.shape
        self.dithered_image = np.zeros_like(image)  # Initialize a blank image for the dithered output

    def floyd_steinberg_dithering(self):
        # Implement Floyd-Steinberg dithering algorithm
        for row in range(self.rows):
            for col in range(self.columns):
                # Get the original pixel value
                orginal_pixel = self.image[row, col]
                # Determine the updated pixel value (binary)
                updated_pixel = 255 * (orginal_pixel > 128)
                self.dithered_image[row, col] = updated_pixel  # Set the dithered pixel value
                # Calculate the error between the original and updated pixel
                error = orginal_pixel - updated_pixel

                # Distribute the error to neighboring pixels
                if col + 1 < self.columns:
                    self.image[row, col + 1] += error * 7 / 16
                if col - 1 >= 0 and row + 1 < self.rows:
                    self.image[row + 1, col - 1] += error * 3 / 16
                if row + 1 < self.rows:
                    self.image[row + 1, col] += error * 5 / 16
                if col + 1 < self.columns and row + 1 < self.rows:
                    self.image[row + 1, col + 1] += error * 1 / 16
        return self.dithered_image.astype(np.uint8)  # unsigned 8-bit integer

    def jarvis_judice_ninke_dithering(self):
        for row in range(self.rows):
            for col in range(self.columns):
                orginal_pixel = self.image[row, col]
                # Determine the updated pixel value (binary)
                updated_pixel = 255 * (orginal_pixel > 128)
                self.dithered_image[row, col] = updated_pixel  # Set the dithered pixel value
                # Calculate the error between the original and updated pixel
                error = orginal_pixel - updated_pixel

                # Distribute the error to neighboring pixels
                if col + 1 < self.columns:
                    self.image[row, col + 1] += error * 7 / 48
                if col + 2 < self.columns:
                    self.image[row, col + 2] += error * 5 / 48
                if col - 2 >= 0 and row + 1 < self.rows:
                    self.image[row + 1, col - 2] += error * 3 / 48
                if col - 1 >= 0 and row + 1 < self.rows:
                    self.image[row + 1, col - 1] += error * 5 / 48
                if row + 1 < self.rows:
                    self.image[row + 1, col] += error * 7 / 48
                if col + 1 < self.columns and row + 1 < self.rows:
                    self.image[row + 1, col + 1] += error * 3 / 48
                if col + 2 < self.columns and row + 1 < self.rows:
                    self.image[row + 1, col + 2] += error * 1 / 48
        return self.dithered_image.astype(np.uint8)  # unsigned 8-bit integer

# Apply Floyd-Steinberg dithering
print("Dithering floyd steinberg ........")
fsd = Dithering(gray_image.copy())
floyd_dithered = fsd.floyd_steinberg_dithering()
cv2.imwrite('Q2_3_floyd_dithered_image.jpg', floyd_dithered)
print("Q2_floyd_dithered_image.jpg Saved")

# Apply Jarvis, Judice, and Ninke dithering
print("Dithering jarvis judice ninke ........")
jjnd = Dithering(gray_image.copy())
jarvis_judice_ninke = jjnd.jarvis_judice_ninke_dithering()
cv2.imwrite('Q2_4_jarvis_judice_ninke_image.jpg', jarvis_judice_ninke)
print("Q2_jarvis_judice_ninke_image.jpg Saved")
