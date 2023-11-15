import os
import cv2

def batch_chronological_filenaming(file_dir):
    try:
        if os.path.exists(file_dir):
            files = os.listdir(file_dir)
            png_files = [f for f in files if f.lower().endswith('.png')]
            png_files.sort()
            counter = 1
            for filename in png_files:
                new_filename = f'{counter:05d}.png'
                old_path = os.path.join(file_dir, filename)
                new_path = os.path.join(file_dir, new_filename)
                os.rename(old_path, new_path)
                counter += 1
            return True
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    return False


def batch_image_resampling(input_dir, output_dir, resampling_rate):
    try:
        if os.path.exists(input_dir):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            files = os.listdir(input_dir)
            png_files = [f for f in files if f.lower().endswith('.png')]
            png_files.sort()
            counter = 1

            for filename in png_files:
                img = cv2.imread(os.path.join(input_dir, filename))
                resized_img = cv2.resize(img, (int(img.shape[1] * resampling_rate), int(img.shape[0] * resampling_rate)))
                cv2.imwrite(os.path.join(output_dir, filename), resized_img)

                counter += 1
            return True
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    return False

def rename_images(directory, prepend_str):
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):  # Check if the file is a PNG image
            new_filename = prepend_str + filename
            original_filepath = os.path.join(directory, filename)
            new_filepath = os.path.join(directory, new_filename)
            os.rename(original_filepath, new_filepath)
            print(f"Renamed '{filename}' to '{new_filename}'")

# Example usage:
# rename_images('data/processed/source-ik/beginner', 'beginner-')

def validate_images_and_labels(source_directory, source_masterlist):
    labels = source_masterlist['filename'].tolist()
    images = []
    no_match = []
    
    for image in os.listdir(source_directory):
        if image.endswith('.png'):
            filename = os.path.basename(image)
            images.append(os.path.splitext(filename)[0])
        
    for label in labels:
        if label not in images:
            print(label, " has no matching image")
            no_match.append(label)

    if len(no_match) == 0:
        print(f'All images and labels matched for {len(images)} items')
        return True
    else:
        return sorted(no_match)