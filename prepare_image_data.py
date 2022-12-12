import boto3
import os

s3_client = boto3.client('s3')

bucket_name = 'aicore-airbnb-images'
folder_path = 'C:\Users\lcox1\Documents\VSCode\AiCore\Data science\images'

for file_name in os.listdr(folder_path):
    file_path = os.path.join(folder_path , file_name)

    key = file_name

    s3_client.upload_file(file_path, bucket_name, key)

