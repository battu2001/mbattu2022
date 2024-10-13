import cv2
import numpy as np

# Read the input image
image = cv2.imread('Q4_1_Input.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Save the grayscale image
cv2.imwrite('Q4_2_gray_image.jpg', gray_image)
print("Q4_gray_image.jpg Saved")

class fourier_transform:
    def __init__(self, image):
        self.image = np.array(image, dtype=np.float64)
        self.rows, self.columns = image.shape
        # Compute the 2D Fourier transform and shift the zero frequency component to the center
        self.ffts = np.fft.fftshift(np.fft.fft2(self.image))
        # Create meshgrid for distance calculation
        X, Y = np.meshgrid(np.arange(self.columns), np.arange(self.rows))
        # Calculate the distance from the center of the frequency domain
        self.D = np.sqrt((X - self.columns / 2) ** 2 + (Y - self.rows / 2) ** 2)

    def _inverse_transform(self,filter):
            filter = self.ffts * filter
            # Shift the zero frequency component back
            filter = np.fft.ifftshift(filter)
            # Compute the inverse Fourier transform
            filter = np.fft.ifft2(filter)
            return np.abs(filter)

    def magnitude_spectrum(self):
        # Compute the magnitude of the Fourier transform
        ft_magnitude = np.abs(self.ffts)
        ft_magnitude = np.log1p(ft_magnitude)
        ft_magnitude_normalized = cv2.normalize(ft_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        return np.uint8(ft_magnitude_normalized)

    def apply_butterworth(self,sigma, order=3):
        # Create a Butterworth filter in the frequency domain
        filter = 1 / (1 + (self.D / sigma) ** (2 * order))
        return self._inverse_transform(filter)

    def apply_gaussian(self,sigma):
        # Create a Gaussian filter in the frequency domain
        filter = np.exp(-(self.D ** 2) / (2 * (sigma ** 2)))
        return self._inverse_transform(filter)

ft = fourier_transform(gray_image.copy())
ft_magnitude = ft.magnitude_spectrum()
cv2.imwrite('Q4_3_magnitude_image.jpg', ft_magnitude)
print("Q4_magnitude_image.jpg Saved")

ft_butterworth = ft.apply_butterworth(sigma=30, order=3)
cv2.imwrite('Q4_4_butterworth_image.jpg', ft_butterworth)
print("Q4_butterworth_image.jpg Saved")

ft_gaussian = ft.apply_gaussian(sigma=30)
cv2.imwrite('Q4_5_gaussian_image.jpg', ft_gaussian)
print("Q4_gaussian_image.jpg Saved")
