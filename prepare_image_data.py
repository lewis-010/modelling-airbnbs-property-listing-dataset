import boto3
import os

# Set up the S3 client
s3 = boto3.client('s3', aws_access_key_id='AKIAXKFMUG3C63ZJ344K', aws_secret_access_key='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', region_name='eu-west-2')

# List all the folders in the bucket
response = s3.list_objects_v2(Bucket='aicore-airbnb-images')

folders = []
for item in response['Contents']:
    if item['Key'].endswith('/'):
        folders.append(item['Key'])

# Create the local directory
local_dir = r'C:\Users\lcox1\Documents\VSCode\AiCore\Data science\images'

# Iterate through the folders and download them to the local directory
for folder in folders:
    s3.download_file('aicore-airbnb-images', Key=folder, Filename=local_dir)
