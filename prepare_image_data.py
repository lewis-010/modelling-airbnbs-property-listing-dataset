import boto3

# Set the AWS access key ID and secret access key for your boto3 client
aws_access_key_id = "AKIAXKFMUG3C63ZJ344K"
aws_secret_access_key = "azdTL4zRcDXPSJYRp8jNJ9d6rOuD6ENpBUa/xxLI"

# Set the name of the bucket that you want to download from
bucket_name = "aicore-airbnb-images"

# Create a boto3 client for the S3 service
s3 = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)

# List all objects in the bucket
response = s3.list_objects_v2(Bucket=bucket_name)

# Loop through the list of objects and download each one
for obj in response["Contents"]:
    key = obj["Key"]
    s3.download_file(bucket_name, key, key)
