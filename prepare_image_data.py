import boto3
import os
from PIL import Image

def download_images():
# Set up the S3 client
    s3 = boto3.client('s3')

    # List all the folders in the bucket
    response = s3.list_objects_v2(Bucket='aicore-airbnb-project')

    folders = []
    for item in response['Contents']:
        if item['Key'].endswith('/'):
            folders.append(item['Key'])

    # Create the local directory
    local_dir = r'C:\Users\lcox1\Documents\VSCode\AiCore\Data science\images'

    # Iterate through the folders and download them to the local directory
    for folder in folders:
        s3.download_file('aicore-airbnb-images', Key=folder, Filename=local_dir)


def resize_images():
    # Set the base directory where the subdirectories are located
    base_dir = r'C:\Users\lcox1\Documents\VSCode\AiCore\Data science\images'

    rgb_file_paths = []

    # Iterate through the subdirectories
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            # Iterate through the files in the subdirectory
            for f in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, f)
                if os.path.isfile(file_path):
                    # Open the image and check its format
                    with Image.open(file_path) as img:
                        if img.mode == 'RGB':
                            # Handle non-RGB images as needed
                            rgb_file_paths.append(file_path)
    
    # Find the height of the smallest PNG
    min_height = float('inf')
    for checked_file in rgb_file_paths:
        with Image.open(checked_file) as im:
            min_height = min(min_height, im.height)

    # Iterate through the file paths, resize the images and save them
    for file_path in rgb_file_paths:
        with Image.open(file_path) as im:
            # Calculate the new height, preserving the aspect ratio
            width, height = im.size
            new_height = min_height
            new_width = int(width * new_height / height)

            # Resize the image
            resized_im = im.resize((new_width, new_height))

            # Save the resized image to the processed_images directory
            resized_im.save(os.path.join('processed_images', os.path.basename(file_path)))
    

                        
