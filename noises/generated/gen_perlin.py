import os
import numpy as np
from PIL import Image
from opensimplex import OpenSimplex
from multiprocessing import Pool, cpu_count

# Parameters for noise generation
octaves = 6  # More octaves increase detail
#persist = 0.5  # Persistence affects amplitude of octaves
sizes = [256]  # Image sizes
num_images = 418560

def generate_perlin_noise_2d(shape, res, seed):
    gen = OpenSimplex(seed=seed)

    noise = np.zeros(shape)
    frequency = res[0] / shape[0]
    amplitude = 1

    for _ in range(octaves):
        for dy in range(shape[0]):
            for dx in range(shape[1]):
                x = frequency * dx
                y = frequency * dy
                noise[dy][dx] += amplitude * gen.noise2(x, y)
        frequency *= 2
        amplitude *= np.random.uniform(0.0, 0.7)

    return (noise - np.min(noise)) / np.ptp(noise)  # Normalize to [0, 1]

def generate_image_perlin_grayscale(params):
    size, i = params
    # Create directories if they do not exist
    folder_grayscale = f'persist-0.0-0.7-new_perlin_grayscale/{size}x{size}'
    os.makedirs(folder_grayscale, exist_ok=True)
    noise = generate_perlin_noise_2d((size, size), (np.random.uniform(100,150), np.random.uniform(100,150)), i)
    im_gray = Image.fromarray(np.uint8(noise * 255), 'L')  # Convert to 8-bit pixel values
    im_gray.save(os.path.join(folder_grayscale, f'noise_{i}.jpg'))

def generate_image_perlin_rgb(params):
    size, i = params
    # Create directories if they do not exist
    folder_rgb = f'perlin_rgb/{size}x{size}'
    os.makedirs(folder_rgb, exist_ok=True)

    # RGB (separate generation for red, green and blue channels)
    noise_r = generate_perlin_noise_2d((size, size), (np.random.uniform(100,150), np.random.uniform(100,150)), i+1000)
    noise_g = generate_perlin_noise_2d((size, size), (np.random.uniform(100,150), np.random.uniform(100,150)), i+2000)
    noise_b = generate_perlin_noise_2d((size, size), (np.random.uniform(100,150), np.random.uniform(100,150)), i+3000)
    im_rgb = Image.merge("RGB", (Image.fromarray(np.uint8(noise_r * 255)), Image.fromarray(np.uint8(noise_g * 255)), Image.fromarray(np.uint8(noise_b * 255))))
    im_rgb.save(os.path.join(folder_rgb, f'noise_{i}.jpg'))

def generate_image_perlin_single_channel(params):
    size, i, color = params
    color_dict = {'red': 0, 'green': 1, 'blue': 2}
    color_index = color_dict[color]

    # Create directories if they do not exist
    folder_color = f'perlin_{color}/{size}x{size}'
    os.makedirs(folder_color, exist_ok=True)

    # Create a blank RGB image
    im = np.zeros((size, size, 3), dtype=np.uint8)

    # Add noise to the selected color channel
    noise = generate_perlin_noise_2d((size, size), (np.random.uniform(100,150), np.random.uniform(100,150)), i+1000*color_index)
    im[..., color_index] = np.uint8(noise * 255)

    # Save the image
    im_color = Image.fromarray(im, 'RGB')
    im_color.save(os.path.join(folder_color, f'noise_{i}.png'))

def generate_all_perlin():
    # Prepare arguments for multiprocessing
    args_grayscale = [(size, i) for size in sizes for i in range(num_images)]
    #args_rgb = [(size, i) for size in sizes for i in range(num_images)]
    #args_single_channel = [(size, i, color) for size in sizes for i in range(num_images) for color in ['red', 'green', 'blue']]

    # Create a pool of workers and map the function to the inputs
    with Pool(cpu_count()) as p:
        p.map(generate_image_perlin_grayscale, args_grayscale)
        #p.map(generate_image_perlin_rgb, args_rgb)
        #p.map(generate_image_perlin_single_channel, args_single_channel)


if __name__ == '__main__':
    generate_all_perlin()