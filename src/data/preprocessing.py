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