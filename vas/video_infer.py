"""Run inference directly from source videos."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch

from .config import Config
from .dataset import SessionData, ImageTransform
from .visualize import save_timeseries_plot

logger = logging.getLogger(__name__)


import csv
from datetime import datetime

@dataclass
class VideoInfo:
    session_id: str
    subject_id: int
    date: str
    condition: str
    trial: str
    video_path: Optional[str] = None


def load_experiment_csv(csv_path: str) -> Dict[str, str]:
    """Load experiment CSV and return a mapping of session_id -> video_path."""
    mapping = {}
    path = Path(csv_path)
    if not path.exists():
        logger.warning("Experiment CSV not found: %s", csv_path)
        return mapping

    with path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Expected columns: session_id, video_path, ...
            # Adapt column names as needed based on the user's CSV structure description
            # "experiment_time_only_sapporo.csv" likely has session_id and file path.
            # Let's assume some flexibility or standard naming.
            # If the user said: "experiment_time_only_sapporo.csv"
            # We'll assume columns "session_id" and "video_path" or similar.
            # Since we don't know the exact columns, let's look for "session_id" and "video_path".
            # Or iterate and try to find logical columns.
            
            # User provided example path: /home/user/alcohol_exp/workspace/vas_detection/experiment_time_only_sapporo.csv
            # We'll assume it has 'session_id' and 'video_path' or 'file_path'
            
            sid = row.get("session_id") or row.get("SessionID")
            vpath = row.get("video_path") or row.get("FilePath") or row.get("path")
            
            if sid and vpath:
                mapping[sid.strip()] = vpath.strip()
    return mapping

import os


class MediaPipeFaceDetector:
    """MediaPipe face detector wrapper (same as training)."""
    
    def __init__(self, mp_face_detection):
        self._detector = mp_face_detection.FaceDetection(
            model_selection=1,  # Full-range model
            min_detection_confidence=0.5,
        )
        self._closed = False
    
    def process(self, frame_rgb):
        """Run face detection on RGB frame."""
        if self._closed:
            return None
        return self._detector.process(frame_rgb)
    
    def close(self):
        if not self._closed and self._detector is not None:
            self._detector.close()
        self._closed = True



import os
import urllib.request

class MediaPipeTasksFaceDetector:
    """Wrapper for MediaPipe Tasks API (Face Detector)."""
    
    def __init__(self, model_path: str):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.5
        )
        self.detector = vision.FaceDetector.create_from_options(options)
        self._closed = False
        
    def process(self, frame_rgb):
        """Run face detection using Tasks API and convert to legacy format."""
        if self._closed:
            return None
            
        import mediapipe as mp
        try:
            # Convert numpy array to MediaPipe Image
            # Ensure contiguous array
            if not frame_rgb.flags['C_CONTIGUOUS']:
                frame_rgb = np.ascontiguousarray(frame_rgb)
                
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            # Detect
            detection_result = self.detector.detect(mp_image)
            
            if not detection_result.detections:
                logger.debug("Tasks API: No face detected.")
                return type('Result', (), {'detections': []})()
                
            # Convert format to match legacy Solutions API
            H, W = frame_rgb.shape[:2]
            
            legacy_detections = []
            for det in detection_result.detections:
                bbox = det.bounding_box
                # Normalize to [0, 1] as expected by the rest of the code
                norm_box = type('RelativeBoundingBox', (), {
                    'xmin': bbox.origin_x / W,
                    'ymin': bbox.origin_y / H,
                    'width': bbox.width / W,
                    'height': bbox.height / H,
                })()
                
                loc_data = type('LocationData', (), {'relative_bounding_box': norm_box})()
                legacy_det = type('Detection', (), {'location_data': loc_data})()
                legacy_detections.append(legacy_det)
                
            return type('Result', (), {'detections': legacy_detections})()
        except Exception as e:
            logger.warning("Tasks API process failed: %s", e)
            return type('Result', (), {'detections': []})()

    def close(self):
        if not self._closed:
            self.detector.close()
        self._closed = True


class MediaPipeSolutionsFaceDetector:
    """Wrapper for MediaPipe Solutions API (Legacy)."""
    def __init__(self, mp_face_detection):
        self._detector = mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        self._closed = False
    
    def process(self, frame_rgb):
        if self._closed: return None
        return self._detector.process(frame_rgb)
    
    def close(self):
        if not self._closed: self._detector.close()
        self._closed = True


def _download_model_if_needed(model_name="blaze_face_short_range.tflite"):
    """Download MediaPipe face detection model if not present."""
    url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    
    # Save to user's cache dir or local dir
    cache_dir = Path.home() / ".cache" / "mediapipe_models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / model_name
    
    if not model_path.exists():
        logger.info("Downloading MediaPipe model to %s...", model_path)
        try:
            urllib.request.urlretrieve(url, model_path)
            logger.info("Download complete.")
        except Exception as e:
            logger.error("Failed to download model: %s", e)
            return None
            
    return str(model_path)


def _load_face_detector():
    """Load face detector (Solutions API -> Tasks API)."""
    
    # 1. Try MediaPipe Solutions API (Preferred if working)
    try:
        # Save original CUDA setting
        original_cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        os.environ['CUDA_VISIBLE_DEVICES'] = '' # Force CPU to avoid conflict
        
        import mediapipe as mp
        
        if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_detection'):
             detector = MediaPipeSolutionsFaceDetector(mp.solutions.face_detection)
             logger.info("Loaded MediaPipe face detector (Solutions API)")
             
             # Restore CUDA
             if original_cuda_devices is None and 'CUDA_VISIBLE_DEVICES' in os.environ:
                 del os.environ['CUDA_VISIBLE_DEVICES']
             elif original_cuda_devices is not None:
                 os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_devices
             return detector

        # Restore CUDA if failed
        if original_cuda_devices is None and 'CUDA_VISIBLE_DEVICES' in os.environ:
             del os.environ['CUDA_VISIBLE_DEVICES']
        elif original_cuda_devices is not None:
             os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_devices
             
    except Exception as e:
        logger.debug("MediaPipe Solutions API load failed: %s", e)
        # Restore CUDA
        original_cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None) # Re-read just in case
        if 'CUDA_VISIBLE_DEVICES' in os.environ: # Clean up if we set it locally
             pass # Actually we need correct logic here, simplification:
    
    # Ensure env is clean
    if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES'] == '':
         del os.environ['CUDA_VISIBLE_DEVICES']


    # 2. Try MediaPipe Tasks API (New API - for broken installs)
    try:
        logger.info("Solutions API missing. Attempting MediaPipe Tasks API...")
        import mediapipe as mp
        # Check if tasks module exists
        if not hasattr(mp, 'tasks'):
             logger.error("MediaPipe Tasks API also missing.")
             return None
             
        model_path = _download_model_if_needed()
        if not model_path:
             logger.error("Could not download model for Tasks API.")
             return None
             
        detector = MediaPipeTasksFaceDetector(model_path)
        logger.info("Loaded MediaPipe face detector (Tasks API)")
        return detector
        
    except Exception as e:
        logger.error("Failed to load MediaPipe Tasks API: %s", e)
        return None


def parse_session_id(session_id: str) -> Optional[VideoInfo]:
    """Parse session ID to extract video information.
    
    Expected format: [subj]subjectID_date_condition_trial
    Example: 
      1_20230101_alcohol_1
      subj121_20250801_本試験01
    """
    # More flexible pattern: optional 'subj' or 'sub' prefix, then digits
    # Then date, then rest
    pattern = r"^(?:sub|subj)?(\d+)_(\d{8})_(.+)$"
    match = re.match(pattern, session_id)
    if not match:
        logger.debug("Could not parse session_id: %s", session_id)
        return None
    
    subject_id = int(match.group(1))
    date = match.group(2)
    rest = match.group(3)
    
    # Try to extract trial if possible (digits at end)
    trial = "1"
    match_trial = re.search(r"(\d+)$", rest)
    if match_trial:
        trial = match_trial.group(1)
        condition = rest[:match_trial.start()].rstrip('_')
    else:
        condition = rest

    return VideoInfo(
        session_id=session_id,
        subject_id=subject_id,
        date=date,
        condition=condition,
        trial=trial,
    )


def find_video_file(video_root: str, info: VideoInfo) -> Optional[str]:
    """Search for video file matching the session info recursively.
    
    Target format based on user report:
    Directory: {date}_.../カメラデータ/
    File: {yymmdd}_{subj_id}.mp4 (e.g., 241015_111.mp4)
    Also supports standard patterns.
    """
    root = Path(video_root)
    if not root.exists():
        return None
        
    # Prepare search patterns
    # pattern 1: {yymmdd}_{subj_id}.mp4 (e.g. 241015_111.mp4)
    yymmdd = info.date[2:] if len(info.date) == 8 else info.date
    target_name_1 = f"{yymmdd}_{info.subject_id}.mp4"
    
    # pattern 2: {date}_{subj_id}.mp4 (e.g. 20241015_111.mp4)
    target_name_2 = f"{info.date}_{info.subject_id}.mp4"
    
    # pattern 3: {subj_id}_{date}.mp4
    target_name_3 = f"{info.subject_id}_{info.date}.mp4"

    # Recursive search
    try:
        # Try finding by filename first (fastest and most specific)
        candidates = list(root.rglob(target_name_1))
        if candidates:
            return str(candidates[0])
            
        candidates = list(root.rglob(target_name_2))
        if candidates:
            return str(candidates[0])
            
        candidates = list(root.rglob(target_name_3))
        if candidates:
            return str(candidates[0])
            
        # Try finding roughly
        # {subj_id} and {date} in filename
        for ext in [".mp4", ".avi", ".mov"]:
            candidates = list(root.rglob(f"*{info.subject_id}*{yymmdd}*{ext}"))
            if candidates:
                return str(candidates[0])

    except Exception as e:
        logger.debug("Error during recursive search: %s", e)
        
    return None


def _video_duration(cap: cv2.VideoCapture) -> float:
    """Get video duration in seconds."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps > 0:
        return frame_count / fps
    return 0.0


def _extract_frame(
    cap: cv2.VideoCapture,
    t_sec: float,
    detector,
    transform,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Extract and preprocess a single frame from video at time t_sec.
    
    Uses the same face detection and cropping logic as the training code.
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        logger.warning("Invalid FPS: %s", fps)
        return None
    
    frame_idx = int(t_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        logger.warning("Could not read frame at t=%.2f", t_sec)
        return None
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect face and crop using MediaPipe (same as training)
    if detector is not None:
        try:
            results = detector.process(frame_rgb)
            if results and results.detections:
                det = results.detections[0]
                bbox = det.location_data.relative_bounding_box
                h, w, _ = frame.shape
                
                x1 = max(int(bbox.xmin * w), 0)
                y1 = max(int(bbox.ymin * h), 0)
                x2 = min(x1 + int(bbox.width * w), w)
                y2 = min(y1 + int(bbox.height * h), h)
                
                if x1 < x2 and y1 < y2:
                    # Crop and resize to 224x224 (same as training)
                    face_crop = frame_rgb[y1:y2, x1:x2]
                    face_crop = cv2.resize(face_crop, (224, 224))
                    frame_rgb = face_crop
                else:
                    logger.warning("Invalid face bbox at t=%.2fs: %s", t_sec, bbox)
                    return None
            else:
                logger.debug("No face detected at t=%.2fs", t_sec)
                # Save debug image
                # cv2.imwrite(f"debug_no_face_{t_sec}.jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                return None
        except Exception as e:
            logger.debug("Face detection failed at t=%.2fs: %s", t_sec, e)
            return None
    
    # Apply transform
    try:
        # Convert numpy (H, W, C) -> Tensor (C, H, W)
        tensor_img = torch.from_numpy(frame_rgb).permute(2, 0, 1)
        tensor = transform(tensor_img)
        return tensor.unsqueeze(0).to(device)
    except Exception as e:
        logger.warning("Transform failed: %s", e)
        return None


def run_video_visualization(
    cfg: Config,
    output_dir: str,
    sessions: Dict[str, SessionData],
    session_ids: Iterable[str],
    video_root: str,
    model: torch.nn.Module,
    device: torch.device,
    experiment_csv: Optional[str] = None,
) -> None:
    detector = _load_face_detector()
    if detector is None:
        logger.error("Face detector is required for video inference. Please check OpenCV installation.")
        return

    csv_mapping = {}
    if experiment_csv:
        csv_mapping = load_experiment_csv(experiment_csv)
        logger.info("Loaded %d video paths from CSV", len(csv_mapping))

    transform = ImageTransform(cfg, train=False)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session_list = list(session_ids)
    for idx, session_id in enumerate(session_list, 1):
        session = sessions.get(session_id)
        if session is None:
            logger.warning("Session not found in index: %s", session_id)
            continue
        
        video_path = None
        if session_id in csv_mapping:
            video_path = csv_mapping[session_id]
        else:
            # Fallback to parsing/searching logic
            info = parse_session_id(session_id)
            if info:
                video_path = find_video_file(video_root, info)
        
        if not video_path:
            logger.warning("Video not found for %s", session_id)
            continue

        # Convert relative path in CSV to absolute if needed
        if not Path(video_path).is_absolute() and video_root:
             video_path = str(Path(video_root) / video_path)

        if not Path(video_path).exists():
             logger.warning("Video file does not exist: %s", video_path)
             continue

        logger.info("Processing %s: %s", session_id, video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning("Failed to open video: %s", video_path)
            continue

        duration = _video_duration(cap)
        if duration <= cfg.clip_sec:
            cap.release()
            logger.warning("Video too short: %s", video_path)
            continue

        anchor_tensor = _extract_frame(
            cap,
            t_sec=cfg.clip_sec / 2.0, # Middle of anchor window
            detector=detector,
            transform=transform,
            device=device,
        )
        if anchor_tensor is None:
            cap.release()
            logger.debug("Anchor extraction failed: %s", session_id)
            continue

        times = []
        probs = []
        t = 0.0
        model.eval()
        
        # Use configurable stride
        stride = cfg.infer_stride_sec
        if stride <= 0:
            stride = 10  # Default fallback if config is 0/invalid

        with torch.no_grad():
            while t + cfg.clip_sec <= duration:
                # Sample middle frame of current window
                mid_t = t + cfg.clip_sec / 2.0
                target_tensor = _extract_frame(
                    cap,
                    t_sec=mid_t,
                    detector=detector,
                    transform=transform,
                    device=device,
                )
                if target_tensor is None:
                    t += stride
                    continue
                logits = model(anchor_tensor, target_tensor)
                prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
                times.append(mid_t)
                probs.append(prob)
                t += stride

        cap.release()

        if probs:
            data = {"times": np.array(times), "probs": np.stack(probs, axis=0)}
            save_timeseries_plot(out_dir, session, data, cfg)
            logger.info("Video viz %s/%s done: %s", idx, len(session_list), session_id)

    if detector is not None:
        detector.close()
