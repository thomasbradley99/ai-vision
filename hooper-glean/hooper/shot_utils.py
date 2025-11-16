import cv2
import math
import numpy as np
import numpy.typing as npt
from typing import Optional, List, Dict, Tuple
from scipy.spatial import ConvexHull
from scipy.ndimage import label, generate_binary_structure
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from scipy.ndimage import center_of_mass


def find_component_clusters(
    labeled_mask: npt.NDArray,
    num_components: int,
    max_split_distance: float,
) -> Dict[int, List[int]]:
    """Find clusters of nearby components that might represent a split ball.
    @param labeled_mask: Mask where each component has a unique integer label
    @param num_components: Number of components in the mask
    @param max_split_distance: Maximum distance between components to be considered part of same cluster
    @return Dictionary mapping cluster IDs to lists of component IDs
    """
    # Get centroids of all components
    centers = center_of_mass(labeled_mask, labeled_mask, range(1, num_components + 1))
    centroids = []
    component_ids = []
    for i, center in enumerate(centers, 1):
        if not np.isnan(center[0]):
            centroids.append(center)
            component_ids.append(i)

    if len(centroids) == 0:
        return {}

    # Use DBSCAN to cluster nearby components
    clustering = DBSCAN(eps=max_split_distance, min_samples=1).fit(centroids)

    # Group component IDs by cluster
    clusters = {}
    for cluster_id, component_idx in zip(clustering.labels_, range(len(component_ids))):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(component_ids[component_idx])

    return clusters


def compute_density_score(coords: npt.NDArray) -> Tuple[float, float]:
    """Compute a score based on pixel density."""
    if len(coords) == 0:
        return 0.0, 0.0

    coords = np.array(coords)

    # Calculate centroid
    centroid = coords.mean(axis=0)

    # Calculate distances from centroid to all pixels
    distances = cdist([centroid], coords)[0]

    # Find the effective radius of the component (max distance from centroid)
    effective_radius = max(distances)

    if effective_radius == 0:
        return 0.0, 0.0

    # Count actual pixels
    actual_area = len(coords)

    # Calculate base density score using convex hull
    try:
        hull = ConvexHull(coords)
        if hull.area > 0:
            density_score = min(1.0, actual_area / hull.area)
        else:
            density_score = 0.0
    except Exception:
        density_score = 0.0

    return density_score, effective_radius


def compute_circularity_score(coords: npt.NDArray, fill_hull: bool = False) -> float:
    """Calculate circularity score for a set of coordinates.
    Returns a value between 0 and 1, where 1 is a perfect circle.
    @param fill_hull (bool): If we should fill in the convex hull when computing.
    """
    # Create binary mask of the component/cluster
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    mask = np.zeros((y_max - y_min + 3, x_max - x_min + 3), dtype=np.uint8)
    coords_normalized = coords - [x_min - 1, y_min - 1]
    mask[coords_normalized[:, 1], coords_normalized[:, 0]] = 1

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    if fill_hull:
        # Create a convex hull of ALL points, not just contours
        hull_points = cv2.convexHull(coords_normalized)

        # Create an empty mask and fill the hull
        hull_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.fillPoly(hull_mask, [hull_points], 1)  # type: ignore

        # Find contour of the filled hull
        contours, _ = cv2.findContours(
            hull_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0.0

        # Use the hull contour for calculations
        contour = contours[0]  # Should be only one contour
    else:
        # Use the largest contour
        contour = max(contours, key=cv2.contourArea)

    # Calculate perimeter and area
    perimeter = cv2.arcLength(contour, True)
    actual_area = cv2.contourArea(contour)

    if perimeter > 0:
        circularity = (4 * np.pi * actual_area) / (perimeter * perimeter)
        return min(1.0, circularity)  # Cap at 1.0

    return 0.0


def compute_score(coords: npt.NDArray, ball_radius: int = 15) -> float:
    """Overall helper function for score computation."""
    density_score, effective_radius = compute_density_score(coords)
    circularity_score = compute_circularity_score(coords)
    # sqrt downweights the circularity score
    score = density_score * math.sqrt(circularity_score)
    # Penalty for components that are too small relative to expected ball size
    if effective_radius < ball_radius / 2:
        # Do not zero out small because we may just be missing a portion of the ball
        score *= effective_radius / (ball_radius / 2)
    # Penalty for components that are too large
    if effective_radius > ball_radius * 1.5:
        # For big balls, we must zero since we know its not right
        score = 0
    return score


def compute_score_from_mask(
    mask: npt.NDArray,
    timestamp: float,
    ball_radius: int = 15,
    max_split_distance: Optional[float] = None,
    indiv_components: bool = False,
) -> Tuple[npt.NDArray, Dict, Dict]:
    """Compute ball scores for components and clusters of nearby components.
    Scores are normalized between 0 and 1, where 1 represents a perfect circular cluster.

    @param mask: Binary mask of motion detection
    @param ball_radius: Approximate radius of basketball in pixels
    @param max_split_distance: Maximum distance to consider components part of same cluster
                               If None, defaults to ball_radius
    @return labeled_mask: Mask where each component has a unique integer label
    @return density_scores: Dictionary mapping component/cluster IDs to their density scores
    @return clusters: Dictionary mapping cluster IDs to lists of component IDs
    """
    if max_split_distance is None:
        max_split_distance = ball_radius

    # Find connected components
    structure = generate_binary_structure(2, 2)  # 8-connectivity
    labeled_mask, num_components = label(mask, structure)  # type: ignore

    # Find clusters of nearby components
    clusters = find_component_clusters(labeled_mask, num_components, max_split_distance)

    scores = {}

    # Get all non-zero coordinates and their labels in one operation
    coords_y, coords_x = np.nonzero(labeled_mask)
    labels = labeled_mask[coords_y, coords_x]
    all_coords = np.column_stack([coords_y, coords_x])

    # Compute scores for individual components
    if indiv_components:
        for comp_id in range(1, num_components + 1):
            comp_mask = labels == comp_id
            if np.any(comp_mask):
                comp_coords = all_coords[comp_mask]
                scores[f"component_{comp_id}"] = compute_score(
                    comp_coords, ball_radius=ball_radius
                )

    # Compute scores for clusters
    for cluster_id, component_ids in clusters.items():
        # Get all coordinates for components in this cluster
        cluster_coords = all_coords[np.isin(labels, component_ids)]
        scores[f"cluster_{cluster_id}"] = compute_score(
            cluster_coords, ball_radius=ball_radius
        )

    return labeled_mask, scores, clusters


def track_components(
    current_mask: npt.NDArray,
    previous_masks: List[npt.NDArray],
    scores: Dict[str, float],
    clusters: Dict[int, List[int]],
    min_score: float = 0.5,
    trajectory_frames: int = 5,
) -> List[str]:
    """Track components and clusters across frames to identify basketball-like motion.
    @param current_mask: Current frame's binary mask
    @param previous_masks: List of previous frame masks
    @param scores: Dictionary of component/cluster density scores
    @param clusters: Dictionary mapping cluster IDs to component IDs
    @param min_score: Minimum density score to consider
    @param trajectory_frames: Number of frames to analyze for trajectory
    @return candidates: List of component/cluster IDs that might be the basketball
    """
    ball_candidates = []

    # Filter by score first
    all_candidates = [obj_id for obj_id, score in scores.items() if score >= min_score]

    if not all_candidates:
        return []

    # Get centroids for tracking
    current_centroids = {}
    labeled_mask, _ = label(current_mask)  # type: ignore

    for obj_id in all_candidates:
        if obj_id.startswith("component_"):
            comp_id = int(obj_id.split("_")[1])
            coords = np.column_stack(np.where(labeled_mask == comp_id))
        else:  # cluster
            cluster_id = int(obj_id.split("_")[1])
            coords = []
            for comp_id in clusters[cluster_id]:
                comp_coords = np.column_stack(np.where(labeled_mask == comp_id))
                coords.extend(comp_coords)
            coords = np.array(coords)

        if len(coords) > 0:
            current_centroids[obj_id] = coords.mean(axis=0)

    # If we have enough previous frames, check trajectories
    if len(previous_masks) >= trajectory_frames:
        # Get historical centroids for trajectory analysis
        historical_centroids = []
        for prev_mask in previous_masks[-trajectory_frames:]:
            prev_labeled, _ = label(prev_mask)  # type: ignore
            coords = np.column_stack(np.where(prev_labeled > 0))
            if len(coords) > 0:
                historical_centroids.append(coords.mean(axis=0))

        if len(historical_centroids) >= 3:
            # Analyze trajectory for parabolic motion
            for obj_id in all_candidates:
                points = historical_centroids + [current_centroids[obj_id]]
                y_coords = [p[0] for p in points]

                # Check if motion is roughly parabolic
                if len(y_coords) >= 4:
                    diffs = np.diff(y_coords)
                    if all(d1 >= d2 for d1, d2 in zip(diffs, diffs[1:])):
                        ball_candidates.append(obj_id)

    # If we couldn't do trajectory analysis, return dense objects
    if not ball_candidates:
        ball_candidates = all_candidates
    return ball_candidates


def detect_shot(
    current_frame_mask: npt.NDArray,
    timestamp: float,
    previous_masks: Optional[List] = None,
    ball_radius: int = 30,
    min_score: float = 0.5,
    trajectory_frames: int = 10,
) -> Tuple[Dict, Dict]:
    """Detect if a basketball shot is occurring.
    @param current_frame_mask: Binary mask of current frame
    @param timestamp: Current frame timestamp - mostly for debugging
    @param previous_masks: List of masks from previous frames
    @param ball_radius: Approximate radius of basketball in pixels
    @param min_score: Used to filter out bad candidates
    @param trajectory_frames: Used for trajectory filtering
    @return output: Contains summary information about the detection
    @return info: Additional information about the detection
    """
    if previous_masks is None:
        previous_masks = []
    # Compute density scores for components and clusters
    _, scores, clusters = compute_score_from_mask(
        current_frame_mask, timestamp, ball_radius=ball_radius, indiv_components=False
    )
    # Track objects across frames
    candidates = track_components(
        current_frame_mask,
        previous_masks,
        scores,
        clusters,
        min_score=min_score,
        trajectory_frames=trajectory_frames,
    )
    # Filter scores to only include candidates
    candidate_scores = {candidate: scores[candidate] for candidate in candidates}
    # Decision logic
    is_shot = len(candidates) > 0
    max_score = max(candidate_scores.values()) if len(candidate_scores) > 0 else 0
    info = {
        "candidates": candidates,
        "scores": candidate_scores,
        "clusters": clusters,
    }
    output = {
        "timestamp": timestamp,
        "is_shot": is_shot,
        "score": max_score,
    }
    return output, info


def interpolate_shots(shots: List[Dict], gap: int = 3) -> List[Dict]:
    """Interpolate shots where gaps of False detections are surrounded by detections.
    @param shots
    @param gap: maximum gap to interpolate
    """
    result = shots.copy()
    i = 0
    while i < len(result):
        # Skip if current entry is a shot
        if result[i]["is_shot"]:
            i += 1
            continue
        # Look ahead to find end of gap
        gap_length = 0
        j = i
        while j < len(result) and not result[j]["is_shot"] and gap_length <= gap:
            gap_length += 1
            j += 1
        # If gap is too long or we hit the end without finding a shot, skip
        if gap_length > gap or j >= len(result) or not result[j]["is_shot"]:
            i += 1
            continue
        # Look back to find start of gap
        start_idx = i - 1
        if start_idx < 0 or not result[start_idx]["is_shot"]:
            i += 1
            continue
        # We found a valid gap to interpolate
        start_score = result[start_idx]["score"]
        end_score = result[j]["score"]
        # Linear interpolation
        for k in range(gap_length):
            t = (k + 1) / (gap_length + 1)  # Interpolation factor
            interp_score = start_score * (1 - t) + end_score * t
            result[i + k]["is_shot"] = True
            result[i + k]["score"] = interp_score
        i = j
    return result


def find_shot_intervals(
    shots: List[Dict], min_contiguous_length: int = 5
) -> List[Tuple[int, float, float]]:
    """Find contiguous intervals of shots that are at least min_length frames long.
    @param data: List of dictionaries with 'timestamp' and 'is_shot' keys
    @param min_length: Minimum number of frames for an interval to be returned
    @return List of tuples (index, start_timestamp, end_timestamp) for each valid interval
    """
    intervals = []
    start_idx = None
    count = 0

    for i, entry in enumerate(shots):
        if entry["is_shot"]:
            # Start new interval
            if start_idx is None:
                start_idx = i
        else:
            # End current interval if it exists
            if start_idx is not None:
                # Check if interval is long enough
                if i - start_idx >= min_contiguous_length:
                    intervals.append(
                        (
                            count,
                            shots[start_idx]["timestamp"],
                            shots[i - 1]["timestamp"],
                        )
                    )
                    count += 1
                start_idx = None

    # Handle case where last interval extends to end of data
    if (start_idx is not None) and (len(shots) - start_idx >= min_contiguous_length):
        intervals.append((count, shots[start_idx]["timestamp"], shots[-1]["timestamp"]))
        count += 1

    return intervals


def merge_nearby_intervals(
    intervals: List[Tuple[int, float, float]], max_gap: float = 0.5
) -> List[Tuple[int, float, float]]:
    """Merge intervals that are within max_gap seconds of each other.

    @param intervals: List of (start_timestamp, end_timestamp) tuples
    @param max_gap: Maximum gap in seconds between intervals to merge
    @return List of merged (start_timestamp, end_timestamp) tuples
    """
    if not intervals:
        return []

    # Sort intervals by start time
    sorted_intervals = sorted(intervals, key=lambda x: x[0])

    merged = []
    _, current_start, current_end = sorted_intervals[0]
    count = 0

    for _, start, end in sorted_intervals[1:]:
        if start - current_end <= max_gap:
            # Merge this interval with current
            current_end = end
        else:
            # Gap too large, start new interval
            merged.append((count, current_start, current_end))
            current_start, current_end = start, end
            count += 1

    # Add final interval
    merged.append((count, current_start, current_end))
    count += 1

    return merged
