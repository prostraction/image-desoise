from multiprocessing import Pool, cpu_count
from PIL import Image
import numpy as np
import os
from scipy.ndimage import gaussian_filter

iterations = 5
sizes = [256]
num_images = 140000

def generate_wavelet_noise_image_grayscale(args):
    size, iterations = args
    image = np.zeros(size)

    scale = 1.0
    for _ in range(iterations):
        noise = np.random.randn(*size)
        smoothed_noise = gaussian_filter(noise, sigma=scale)
        image += smoothed_noise * scale
        scale /= 2

    return image

def generate_wavelet_noise_image_single_channel(args):
    size, iterations, channel = args
    image = np.zeros(size + (3,))

    scale = 1.0
    for _ in range(iterations):
        noise = np.random.randn(*size)
        smoothed_noise = gaussian_filter(noise, sigma=scale)
        image[..., channel] += smoothed_noise * scale
        scale /= 2

    return image

def generate_wavelet_noise_image_rgb(args):
    size, iterations = args
    image = np.zeros(size + (3,))

    scale = 1.0
    for _ in range(iterations):
        for channel in range(3):
            noise = np.random.randn(*size)
            smoothed_noise = gaussian_filter(noise, sigma=scale)
            image[..., channel] += smoothed_noise * scale
        scale /= 2

    return image

def generate_image_wavelet_grayscale(params):
    size, i = params
    folder_wavelet_grayscale = f'wavelet_grayscale/{size}x{size}'
    os.makedirs(folder_wavelet_grayscale, exist_ok=True)
    noise = generate_wavelet_noise_image_grayscale(((size, size), iterations))
    noise = (noise - np.min(noise)) / np.ptp(noise)
    im = Image.fromarray(np.uint8(noise * 255), 'L')
    im.save(os.path.join(folder_wavelet_grayscale, f'noise_{i}.png'))

def generate_image_wavelet_single_channel(params):
    size, i, channel = params
    channel_names = ['red', 'green', 'blue']
    folder_wavelet_single_channel = f'wavelet_{channel_names[channel]}/{size}x{size}'
    os.makedirs(folder_wavelet_single_channel, exist_ok=True)
    noise = generate_wavelet_noise_image_single_channel(((size, size), iterations, channel))
    noise[..., channel] = (noise[..., channel] - np.min(noise[..., channel])) / np.ptp(noise[..., channel])  # Normalize only the channel with noise
    noise[..., [c for c in range(3) if c != channel]] = 0  # Set the other channels to zero
    im = Image.fromarray(np.uint8(noise * 255), 'RGB')
    im.save(os.path.join(folder_wavelet_single_channel, f'noise_{i}.png'))

def generate_image_wavelet_rgb(params):
    size, i = params
    folder_wavelet_rgb = f'wavelet_rgb/{size}x{size}'
    os.makedirs(folder_wavelet_rgb, exist_ok=True)
    noise = generate_wavelet_noise_image_rgb(((size, size), iterations))
    noise = (noise - np.min(noise)) / np.ptp(noise)
    im = Image.fromarray(np.uint8(noise * 255), 'RGB')
    im.save(os.path.join(folder_wavelet_rgb, f'noise_{i}.png'))

def generate_all_wavelet():
    args_grayscale = [(size, i) for size in sizes for i in range(num_images)]
    #args_single_channel = [(size, i, channel) for size in sizes for i in range(num_images) for channel in range(3)]
    #args_rgb = [(size, i) for size in sizes for i in range(num_images)]
    with Pool(cpu_count()) as p:
        p.map(generate_image_wavelet_grayscale, args_grayscale)
        #p.map(generate_image_wavelet_single_channel, args_single_channel)
        #p.map(generate_image_wavelet_rgb, args_rgb)

if __name__ == '__main__':
    generate_all_wavelet()