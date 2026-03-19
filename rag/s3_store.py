"""
Thin wrapper around boto3 for storing and retrieving knowledge-base
source documents in Amazon S3.

Every ingested document is uploaded with metadata (title, doc_type,
source filename) so the object is self-describing without a separate
catalogue.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from config import settings


class S3Store:
    def __init__(self) -> None:
        self._client = boto3.client("s3", region_name=settings.AWS_REGION)
        self._bucket = settings.S3_BUCKET

    # ── Public ────────────────────────────────────────────────────────────────

    def upload(self, local_path: Path, doc_type: str, title: str) -> str:
        """
        Upload local_path to S3 under a UUID-based key and return the key.

        The key layout is  <doc_type>/<uuid><suffix>  so runbooks and
        incidents occupy separate prefixes for easy manual browsing.
        """
        suffix = local_path.suffix or ".txt"
        key = f"{doc_type}/{uuid.uuid4().hex}{suffix}"
        self._client.upload_file(
            str(local_path),
            self._bucket,
            key,
            ExtraArgs={
                "Metadata": {
                    "title": title,
                    "doc_type": doc_type,
                    "source_filename": local_path.name,
                }
            },
        )
        return key

    def download_text(self, s3_key: str) -> str:
        """Return the full text content of an S3 object."""
        obj = self._client.get_object(Bucket=self._bucket, Key=s3_key)
        return obj["Body"].read().decode("utf-8", errors="replace")

    def ensure_bucket_exists(self) -> None:
        """Create the S3 bucket if it does not already exist."""
        try:
            self._client.head_bucket(Bucket=self._bucket)
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code in ("404", "NoSuchBucket"):
                cfg: dict = {}
                if settings.AWS_REGION != "us-east-1":
                    cfg = {
                        "CreateBucketConfiguration": {
                            "LocationConstraint": settings.AWS_REGION
                        }
                    }
                self._client.create_bucket(Bucket=self._bucket, **cfg)
            else:
                raise
