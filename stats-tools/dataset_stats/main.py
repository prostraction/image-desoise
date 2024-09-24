import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from exif import Image as ExifImage

def print_sample_exif(image_path, limit=5):
    """Print EXIF data for a few sample images to inspect the metadata."""
    sample_count = 0
    for root, _, files in os.walk(image_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as img_file:
                    img = ExifImage(img_file)
                    if img.has_exif:
                        print(f"EXIF data for {file}:")
                        for tag in img.list_all():
                            print(f"{tag}: {getattr(img, tag)}")
                        sample_count += 1
                        if sample_count >= limit:
                            return

def get_image_metadata(image_path):
    """Extract ISO, exposure time, and aperture from image EXIF data."""
    try:
        with open(image_path, 'rb') as img_file:
            img = ExifImage(img_file)
            if img.has_exif:
                iso = getattr(img, 'photographic_sensitivity', None)
                exposure_time = getattr(img, 'exposure_time', None)
                aperture = getattr(img, 'f_number', None)
                return iso, exposure_time, aperture
            else:
                return None, None, None
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None, None, None

def process_image(file_path):
    """Process a single image to get its metadata."""
    return file_path, get_image_metadata(file_path)

def main():
    dir_path = "./../../dataset/CameraDatasetJPEG100"
    
    iso_counter = Counter()
    exposure_time_counter = Counter()
    aperture_counter = Counter()

    print("Sample EXIF data from images:")
    print_sample_exif(dir_path)

    image_files = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                image_files.append(file_path)

    print(f"Found {len(image_files)} images to process.")

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, file_path) for file_path in image_files]
        for future in as_completed(futures):
            file_path, (iso, exposure_time, aperture) = future.result()
            if iso is not None:
                iso_counter[iso] += 1
            if exposure_time is not None:
                exposure_time_counter[exposure_time] += 1
            if aperture is not None:
                aperture_counter[aperture] += 1

    print("\nISO Values and Their Counts:")
    if iso_counter:
        for iso, count in iso_counter.items():
            print(f"ISO {iso}: {count} images")
    else:
        print("No ISO values found.")

    print("\nExposure Times and Their Counts:")
    if exposure_time_counter:
        for exposure_time, count in exposure_time_counter.items():
            print(f"Exposure Time {exposure_time}: {count} images")
    else:
        print("No exposure times found.")

    print("\nAperture Values and Their Counts:")
    if aperture_counter:
        for aperture, count in aperture_counter.items():
            print(f"Aperture {aperture}: {count} images")
    else:
        print("No aperture values found.")

if __name__ == "__main__":
    main()
