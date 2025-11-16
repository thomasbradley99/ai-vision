from tqdm import tqdm
from time import time
from os.path import join, isfile
from typing import List, Dict
from firebase_admin import storage
from shared_utils.types import ClusterOutput


def upload_files_to_firebase(
    filenames: List[str], artifact_dir: str, blob_dir: str
) -> Dict[str, str]:
    """Upload a batch of files in a local directory to a cloud directory.
    @param filenames: List of filenames
    @param artifact_dir: Local data directory containing artifacts
    @param blob_dir: Firebase storage directory
    """
    path_to_url: Dict[str, str] = {}
    for filename in filenames:
        local_path = join(artifact_dir, filename)
        if not isfile(local_path):
            continue

        blob_path = join(blob_dir, filename)
        blob_url = upload_file_to_firebase(local_path, blob_path)
        # . -> _
        path_to_url[filename.replace(".", "_")] = blob_url
    return path_to_url


def upload_cluster_images_to_firebase(
    clusters: List[ClusterOutput], artifact_dir: str, blob_dir: str
) -> List[ClusterOutput]:
    """Upload a directory of files to cloud.
    @param clusters: List of cluster data
    @param artifact_dir: Local data directory where cluster images live
    @param blob_dir: Firebase storage directory
    @return clusters: Updated clusters with `image_url` set
    """
    pbar = tqdm(total=len(clusters), desc="uploading clusters")
    for i in range(len(clusters)):
        # upload image to cloud bucket
        bucket = storage.bucket()
        image_name = clusters[i].image_name
        if image_name is None:
            continue
        image_path = join(artifact_dir, "cluster_images", image_name)
        if not isfile(image_path):
            continue
        blob_path: str = f"{blob_dir}/cluster_images/{image_name}"
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(image_path)
        blob.make_public()
        clusters[i].image_url = blob.public_url
        pbar.update()
    pbar.close()
    return clusters


def upload_file_to_firebase(file_path: str, blob_path: str) -> str:
    """Upload a file to firebase.
    @param file_path:
    @param blob_path: Desired blob path
    """
    bucket = storage.bucket()
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(file_path)
    blob.make_public()

    return blob.public_url


def update_session_processing_progress(
    firebase_client,
    session_id: str,
    progress_delta: int | float,
    doc_field: str = "processingProgress",
) -> bool:
    """Update the processing progress for the session document.
    @param session_id: Session identifier
    @param progress_delta: Amount to add to the progress
    @param doc_field: Field for progress in document (default: processingProgress)
    """
    fb_session_ref = firebase_client.collection("Sessions").document(session_id)
    fb_session = fb_session_ref.get()
    if not fb_session.exists:
        return False  # can't find this session
    fb_session_data = fb_session.to_dict()
    curr_progress = fb_session_data.get(doc_field, 0)
    new_progress = min(round(curr_progress + progress_delta), 100)
    try:
        fb_session_ref.update({doc_field: new_progress, "updatedAt": int(time())})
    except Exception:
        return False
    return True


def hardset_session_processing_progress(
    firebase_client,
    session_id: str,
    progress: int,
    doc_field: str = "processingProgress",
) -> bool:
    """Hard set the processing progress for the session document.
    @param session_id: Session identifier
    @param progress: Desired progress
    @param doc_field: Field for progress in document (default: processingProgress)
    :return success:
    """
    fb_session_ref = firebase_client.collection("Sessions").document(session_id)
    fb_session = fb_session_ref.get()
    if not fb_session.exists:
        return False  # can't find this session
    progress = int(min(max(progress, 0), 100))
    try:
        fb_session_ref.update({doc_field: progress, "updatedAt": int(time())})
    except Exception:
        return False
    return True


def update_session_hoop_stats(
    firebase_client,
    session_id: str,
    frac_missing: float = 0,
    frac_moving: float = 0,
) -> bool:
    """Update the processing progress for the session document.
    @param session_id: Session identifier
    @param frac_missing: Fraction of frames with missing hoop (default: 0)
    @param frac_moving: Fraction of frames with moving hoop (default: 0)
    """
    if frac_missing is None or frac_moving is None:
        return False
    fb_session_ref = firebase_client.collection("Sessions").document(session_id)
    fb_session = fb_session_ref.get()
    if not fb_session.exists:
        return False  # can't find this session
    try:
        fb_session_ref.update(
            {
                "hoopFracMissing": float(min(max(frac_missing, 0), 100)),
                "hoopFracMoving": float(min(max(frac_moving, 0), 100)),
                "updatedAt": int(
                    time(),
                ),
            }
        )
    except Exception:
        return False
    return True
