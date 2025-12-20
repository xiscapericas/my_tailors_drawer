"""AWS S3 handler for downloading RADCURE cases."""

import os
import boto3
from typing import List, Optional


class AWSHandler:
    """Handles AWS S3 operations for RADCURE case management."""
    
    def __init__(
        self,
        bucket_name: str,
        aws_folder: str,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        region_name: str = 'eu-west-1'
    ):
        """
        Initialize AWS handler.
        
        Parameters
        ----------
        bucket_name : str
            S3 bucket name
        aws_folder : str
            S3 folder prefix (e.g., 'RADCURE/all_cases/')
        access_key_id : str, optional
            AWS access key ID. If None, uses environment variable.
        secret_access_key : str, optional
            AWS secret access key. If None, uses environment variable.
        region_name : str
            AWS region name
        """
        self.bucket_name = bucket_name
        self.aws_folder = aws_folder.rstrip('/') + '/'
        self.region_name = region_name
        
        # Set environment variables if provided
        if access_key_id:
            os.environ['AWS_ACCESS_KEY_ID'] = access_key_id
        if secret_access_key:
            os.environ['AWS_SECRET_ACCESS_KEY'] = secret_access_key
        os.environ['AWS_DEFAULT_REGION'] = region_name
        
        # Initialize S3 client
        self.s3_client = boto3.client("s3", region_name=region_name)
    
    def list_cases(self) -> List[str]:
        """
        List all RADCURE case IDs available in S3.
        
        Returns
        -------
        List[str]
            List of case IDs (without .zip extension)
        """
        cases_in_path = []
        continuation_token = None
        
        while True:
            kwargs = {
                "Bucket": self.bucket_name,
                "Prefix": self.aws_folder,
                "Delimiter": "/",
            }
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token
            
            response = self.s3_client.list_objects_v2(**kwargs)
            for item in response.get("Contents", []):
                key = item["Key"]
                if key.endswith(".zip"):
                    case_id = key.replace(self.aws_folder, "").replace(".zip", "")
                    cases_in_path.append(case_id)
            
            if not response.get("IsTruncated"):
                break
            continuation_token = response.get("NextContinuationToken")
        
        return cases_in_path
    
    def download_case(self, radcure_case_id: str, local_folder: str) -> str:
        """
        Download a RADCURE case zip file from S3.
        
        Parameters
        ----------
        radcure_case_id : str
            Case ID (e.g., 'RADCURE-0005')
        local_folder : str
            Local folder to save the zip file
        
        Returns
        -------
        str
            Path to downloaded zip file
        """
        os.makedirs(local_folder, exist_ok=True)
        
        key = f"{self.aws_folder.rstrip('/')}/{radcure_case_id}.zip"
        local_path = os.path.join(local_folder, f"{radcure_case_id}.zip")
        
        print(f"Downloading from s3://{self.bucket_name}/{key}")
        self.s3_client.download_file(self.bucket_name, key, local_path)
        print(f"Download complete for: {radcure_case_id}.zip")
        
        return local_path

