import boto3
import os

s3_client = boto3.client('s3')
dir_path = r'C:\Users\lcox1\Documents\VSCode\AiCore\Data science\images' # path containing folders to upload

parent_bucket_name = 'aicore-airbnb-images' # parent bucket to upload the folders to

# iterate through the subdirectories in the directory
for root, dirs, files in os.walk(dir_path):
    
    # name of subdirectory
    subdir_name = os.path.basename(root)
    
    # create new bucket for subdirectory
    subdir_bucket_name = f'{parent_bucket_name}/{subdir_name}'
    s3_client.create_bucket(Bucket = subdir_bucket_name)

    # iterate through files in subdirectories
    for file_name in files:
        file_path = os.path.join(root, file_name)

        key = file_path

        s3_client.upload_file(file_path, bucket_name, key)

