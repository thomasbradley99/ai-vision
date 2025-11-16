from os.path import join, dirname, realpath

GCLOUD_DEFAULT_LOCATION = "us-central1"
FIREBASE_STORAGE_BUCKET = "hooper-ac7b0.appspot.com"
GCLOUD_FIREBASE_PROJECT_ID = "hooper-ac7b0"
GCLOUD_COMPUTE_PROJECT_ID = "hooper-405102"
GCLOUD_DEFAULT_LOCATION = "us-central1"
SERVICE_ACCOUNT_FILE = realpath(
    join(
        dirname(__file__),
        "credentials/hooper-service-account.json",
    )
)
