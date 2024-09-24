import cv2
import os
import concurrent.futures

def denoise_image(input_path, output_path):
    image = cv2.imread(input_path)
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    cv2.imwrite(output_path, denoised)

def bilateral_denoise_image(input_path, output_path):
    image = cv2.imread(input_path)
    denoised = cv2.bilateralFilter(image, d=15, sigmaColor=75, sigmaSpace=75)
    cv2.imwrite(output_path, denoised)

def gaussian_denoise_image(input_path, output_path):
    image = cv2.imread(input_path)
    denoised = cv2.GaussianBlur(image, (5,5), 0)
    cv2.imwrite(output_path, denoised)

def process_image(filename, input_folder, output_folders, methods):
    input_path = os.path.join(input_folder, filename)
    for method, output_folder in zip(methods, output_folders):
        output_path = os.path.join(output_folder, filename)
        method(input_path, output_path)

def denoise_images_in_folder(input_folder, output_folders, max_workers=5):
    methods = [denoise_image, bilateral_denoise_image, gaussian_denoise_image]

    for folder in output_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_image, filename, input_folder, output_folders, methods) 
                   for filename in os.listdir(input_folder) 
                   if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        # Wait for all threads in the pool to complete
        for future in concurrent.futures.as_completed(futures):
            # You can add error handling here if needed
            future.result()

# Define your input and output folders
input_folder = ".\\..\\..\\dataset\\_pres"
nl_means_output_folder = "1demo_nl_means_denoised"
bilateral_output_folder = "1demo_bilateral_denoised"
gaussian_output_folder = "1demo_gaussian_denoised"
output_folders = [nl_means_output_folder, bilateral_output_folder, gaussian_output_folder]

# Denoise images using all methods
denoise_images_in_folder(input_folder, output_folders)
