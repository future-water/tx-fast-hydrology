import boto3
import logging
from pathlib import Path
from botocore.exceptions import BotoCoreError, ClientError
import pandas as pd

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class S3Settings(BaseSettings):
    access_key_id: str = ""
    secret_access_key: str = ""
    bucket_name: str = ""

    class Config:
        env_file = "kisters_water.env"
        env_file_encoding = "utf-8"
        secrets_dir = "/run/secrets"
        # Assuming Pydantic v1 supports these configurations similarly
        extra = "ignore"  # This might need to be adjusted or removed based on actual support
        env_prefix = "S3_"  # Ensure this matches your .env file's variable prefixes


def get_session():
    settings = S3Settings()
    return boto3.Session(
        aws_access_key_id=settings.access_key_id,
        aws_secret_access_key=settings.secret_access_key,
    )


def save_file_from_s3(
    bucket_name: str, object_key: str, local_dir: Path, target: str | Path | None = None
):
    """
    Downloads a file from S3. If `target` is not provided, saves the file to the local path
    equivalent to `object_key`. Creates necessary directories if they don't exist.

    Args:
        bucket_name (str): Name of the S3 bucket.
        object_key (str): Key of the object in the S3 bucket.
        target (str | None): Local file path to save the file. Defaults to `object_key`.

    Returns:
        bool: True if the file was successfully downloaded, False if the object does not exist.
    """
    try:
        # Get S3 client
        s3_client = get_session().client("s3")

        # Determine target path
        if target is None:
            target = local_dir / object_key  # Save to the same path as object_key
        else:
            target = local_dir / target

        # Create necessary directories
        target.parent.mkdir(parents=True, exist_ok=True)

        # Download the file
        s3_client.download_file(bucket_name, object_key, target)
        logger.info(f"File {object_key} successfully downloaded to {target}")
        return True
    except ClientError as e:
        # If a client error is thrown, check if it was a 404 error
        if e.response["Error"]["Code"] == "404":
            logger.error(f"ERROR: {object_key} does not exist in the S3 bucket {bucket_name}")
            return False
        else:
            # For any other ClientError, raise it
            logger.exception("Unexpected error occurred while accessing S3")
            raise e
    except Exception as e:
        # Handle unexpected exceptions
        logger.exception("Unexpected error occurred while saving the file from S3")
        raise e


def file_exists_in_bucket(bucket_name, object_key):
    """
    Check if a specific file exists in an S3 bucket.

    Parameters:
    - bucket_name: The name of the S3 bucket.
    - object_key: The key of the object within the bucket.

    Returns:
    - True if the file exists, False otherwise.
    """
    # Assuming you have a session or client set up
    try:
        s3_client = get_session().client("s3")
        s3_client.head_object(Bucket=bucket_name, Key=object_key)
        logger.debug(f"Validation successful for {object_key}. File exists.")
        return True  # The file exists
    except ClientError as e:
        # If a client error is thrown, check if it was a 404 error
        # which means the object does not exist.
        if e.response["Error"]["Code"] == "404":
            logger.error(f"ERROR: {object_key} does not exist in the S3 bucket")
            return False
        else:
            # For any other ClientError, raise it
            raise e


def upload_file_to_s3(bucket_name: str, s3_key: str, filename: Path):
    """
    Uploads a file to an S3 bucket and handles exceptions.
    """
    try:
        s3_resource = get_session().resource("s3")
        s3_resource.Bucket(bucket_name).upload_file(Filename=str(filename), Key=s3_key)
        logger.debug(f"{s3_key} uploaded successfully")
        return True
    except (BotoCoreError, ClientError) as error:
        logger.error(f"Failed to upload file: {error}")
        raise ConnectionError(
            f"ERROR: Could not upload {filename} to the S3 bucket to {s3_key}: {error}"
        ) from error


def read_csv_from_s3(bucket_name: str, location: str):
    try:
        s3_client = get_session().client("s3")
        obj = s3_client.get_object(Bucket=bucket_name, Key=location)
        df = pd.read_csv(obj["Body"])
    except Exception as e:
        raise ConnectionError(
            f"ERROR: Could not get the file {location} from the S3 bucket: {e}"
        ) from e
    return df
