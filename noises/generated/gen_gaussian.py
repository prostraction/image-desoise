import numpy as np
import cv2

def generate_gaussian_noise(mean, std_dev, size):
    """Generate Gaussian noise for grayscale."""
    return np.random.normal(mean, std_dev, (size, size))

def save_noise_image(noise_image, filename):
    """Save the noise image in 32-bit RGB using OpenCV."""
    # Map the values from [0, 1] to [0, 65535]
    noise_image_mapped = (noise_image * 65535).astype(np.uint16)
    
    # OpenCV uses BGR format, so convert RGB to BGR
    noise_image_bgr = noise_image_mapped[:, :, ::-1]
    cv2.imwrite(filename, noise_image_bgr)

def main():
    # Parameters
    mean = 0.5         # 0.5 for centered noise in the 0-1 range
    std_dev = 0.3      # Adjust as needed
    size = 256
    num_images = 140000
    base_filename = 'gaussian_gs//noise_grayscale_{}.png'

    for i in range(num_images):
        single_channel = generate_gaussian_noise(mean, std_dev, size)
        noise_image = np.stack([single_channel]*3, axis=-1)  # Replicate to RGB
        filename = base_filename.format(i)
        save_noise_image(noise_image, filename)
        print(f"Saved image {i+1}/{num_images}")

if __name__ == "__main__":
    main()
