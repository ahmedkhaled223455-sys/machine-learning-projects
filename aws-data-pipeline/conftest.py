"""Test configuration for the AWS pipeline.

Sets dummy AWS credentials so moto's mocked clients have something to
authenticate with, and makes `src/` importable. Crucially, it clears
AWS_ENDPOINT_URL: moto intercepts boto3 at the botocore layer, so a
lingering LocalStack endpoint from the shell would send calls to a real
socket instead of the mock.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.pop("AWS_ENDPOINT_URL", None)
