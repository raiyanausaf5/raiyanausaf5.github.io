import numpy as np
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt
import os

data_folder = '/Users/raiyanausaf14/Desktop/CS180/project1/data/'
output_folder = '/Users/raiyanausaf14/Desktop/CS180/project1/output/'

#different scoring functions
def process_image(image_path):
    im = skio.imread(image_path)
    im = sk.img_as_float(im)

    # Divide the image into color channels
    height = np.floor(im.shape[0] / 3.0).astype(int)
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    # Normalize brightness
    g_normalized = normalize_brightness(g, b)
    r_normalized = normalize_brightness(r, b)

    # Find the best shifts after normalization using pyramid method
    best_dx_g, best_dy_g = find_best_shift_pyramid(g_normalized, b, max_shift=30, pyramid_levels=4)
    best_dx_r, best_dy_r = find_best_shift_pyramid(r_normalized, b, max_shift=30, pyramid_levels=4)

    # Align the channels
    aligned_g = np.roll(g_normalized, shift=(best_dy_g, best_dx_g), axis=(0, 1))
    aligned_r = np.roll(r_normalized, shift=(best_dy_r, best_dx_r), axis=(0, 1))

    # Stack the channels in RGB order (flip from BGR)
    color_image = np.stack((aligned_r, aligned_g, b), axis=-1)
    
    return color_image

def normalized_crosscorrelation(image1, image2):
    norm1 = np.linalg.norm(image1, 'fro')
    norm2 = np.linalg.norm(image2, 'fro')
    image1 = image1 / norm1
    image2 = image2 / norm2
    return np.sum(image1 * image2)

#shifting images to find the best score
def find_best_shift(channel, reference, max_shift=15):
    best_score = -float('inf')
    best_dx = 0
    best_dy = 0
    
    for dx in range(-max_shift, max_shift + 1):
        for dy in range(-max_shift, max_shift + 1):
            shifted_channel = np.roll(channel, shift=(dy, dx), axis=(0, 1))
            score = normalized_crosscorrelation(reference, shifted_channel)
            
            if score > best_score:
                best_score = score
                best_dx = dx
                best_dy = dy
    
    return best_dx, best_dy

def normalize_brightness(channel, reference):
    ref_mean = np.mean(reference)
    ref_std = np.std(reference)
    
    ch_mean = np.mean(channel)
    ch_std = np.std(channel)
    
    # Normalize the input channel to match the reference channel's mean and std
    normalized_channel = (channel - ch_mean) * (ref_std / ch_std) + ref_mean
    
    return normalized_channel

def gaussian_pyramid(image, levels):
    pyramid = [image]
    for i in range(1, levels):
        image = sk.transform.pyramid_reduce(image, downscale=2)
        pyramid.append(image)
    return pyramid


def find_best_shift_pyramid(channel, reference, max_shift=15, pyramid_levels=4):
    channel_pyramid = gaussian_pyramid(channel, pyramid_levels)
    reference_pyramid = gaussian_pyramid(reference, pyramid_levels)
    
    best_dx, best_dy = 0, 0
    
    for level in range(pyramid_levels - 1, -1, -1):
        best_dx *= 2
        best_dy *= 2
        
        channel_level = np.roll(channel_pyramid[level], shift=(best_dy, best_dx), axis=(0, 1))
        reference_level = reference_pyramid[level]
        
        dx, dy = find_best_shift(channel_level, reference_level, max_shift=max_shift)
        
        best_dx += dx
        best_dy += dy
    
    return best_dx, best_dy

def save_image(output_path, image):
    image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    output_path_jpg = os.path.splitext(output_path)[0] + '.jpg'
    skio.imsave(output_path_jpg, image_uint8)
    print(f"Processed and saved: {output_path_jpg}")

for filename in os.listdir(data_folder):
    image_path = os.path.join(data_folder, filename)
    print(f"Processing: {image_path}")
    
    result_image = process_image(image_path)
    
    # Save the result image
    output_path = os.path.join(output_folder, f'aligned_{filename}')
    save_image(output_path, result_image)

print("All images processed.")