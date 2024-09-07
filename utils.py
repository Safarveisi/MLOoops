import time
from functools import wraps

from dvc.api import DVCFileSystem
import onnxruntime as ort

# URL to the dvc repository
# The public ssh key should be one of the repository's deploy keys
_REPO_URL = "git@github.com:Safarveisi/MLOoops.git"
_FILEPATH = "models/model.onnx"


def get_dvc_file_from_s3(
    file_path: str,
    repo_url:str = _REPO_URL,
    revision: str = "master"
    ) -> ort.InferenceSession:
    with DVCFileSystem(url=repo_url, rev=revision).open(file_path) as f:
        # Get the size in bytes
        file_size_bytes = f.size
        # Convert the file size to MB
        file_size_mb = file_size_bytes / (1024 * 1024)
        # Print the file size in MB
        print(f"File size: {file_size_mb:.2f} MB")
        # Create an ONNX Inference Session
        model = ort.InferenceSession(
            path_or_bytes=f.read(), providers=["CPUExecutionProvider"]
        )

        return model

def timing(f):
    """Decorator for timing functions
    Usage:
    @timing
    def function(a):
        pass
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print("function:%r took: %2.5f sec" % (f.__name__, end - start))
        return result

    return wrapper
