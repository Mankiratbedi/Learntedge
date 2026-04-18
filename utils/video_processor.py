import os
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from moviepy.editor import VideoFileClip


def _detect_face_centers(
    clip: VideoFileClip,
    sample_fps: float = 2.0,
    min_detection_confidence: float = 0.5,
) -> List[Tuple[float, float]]:
    """
    Sample video frames and estimate speaker face centers in normalized coordinates.
    Returns a list of (x, y) values in range [0, 1].
    """
    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=min_detection_confidence,
    )
    centers: List[Tuple[float, float]] = []

    try:
        for frame in clip.iter_frames(fps=sample_fps, dtype="uint8"):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.process(rgb)
            if not result.detections:
                continue

            # Use the first detected face as the speaker candidate.
            bbox = result.detections[0].location_data.relative_bounding_box
            cx = float(bbox.xmin + (bbox.width / 2.0))
            cy = float(bbox.ymin + (bbox.height / 2.0))

            # Clamp values to image bounds in normalized space.
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            centers.append((cx, cy))
    finally:
        detector.close()

    return centers


def _get_target_crop_size(width: int, height: int) -> Tuple[int, int]:
    """
    Compute a maximal 9:16 (vertical) crop that fits in the source frame.
    """
    target_ratio = 9 / 16  # width / height for vertical video
    source_ratio = width / height

    if source_ratio > target_ratio:
        # Too wide: keep full height, trim width.
        crop_h = height
        crop_w = int(round(crop_h * target_ratio))
    else:
        # Too tall/narrow: keep full width, trim height.
        crop_w = width
        crop_h = int(round(crop_w / target_ratio))

    crop_w = max(2, min(crop_w, width))
    crop_h = max(2, min(crop_h, height))
    return crop_w, crop_h


def process_video_clip(
    input_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
) -> str:
    """
    1) Cut the video from start_time to end_time using MoviePy.
    2) Detect speaker face using MediaPipe.
    3) Crop to 9:16 vertical centered on detected face.
    4) Save final result in the output/ folder.

    Returns the absolute path of the saved output file.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if start_time < 0 or end_time <= start_time:
        raise ValueError("Invalid time range. Ensure 0 <= start_time < end_time.")

    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Always write inside output/ regardless of caller's directory input.
    output_filename = os.path.basename(output_path) or "processed_clip.mp4"
    final_output_path = os.path.join(output_dir, output_filename)

    clip: Optional[VideoFileClip] = None
    subclip: Optional[VideoFileClip] = None

    try:
        clip = VideoFileClip(input_path)
        safe_end_time = min(float(end_time), float(clip.duration))
        if safe_end_time <= start_time:
            raise ValueError("Time range exceeds video duration.")

        subclip = clip.subclip(start_time, safe_end_time)
        frame_w, frame_h = int(subclip.w), int(subclip.h)
        crop_w, crop_h = _get_target_crop_size(frame_w, frame_h)

        centers = _detect_face_centers(subclip)
        if centers:
            avg_x = float(np.mean([c[0] for c in centers]))
            avg_y = float(np.mean([c[1] for c in centers]))
        else:
            # Fallback: center frame if face is not detected.
            avg_x, avg_y = 0.5, 0.5

        center_x_px = int(round(avg_x * frame_w))
        center_y_px = int(round(avg_y * frame_h))

        x1 = max(0, min(frame_w - crop_w, center_x_px - (crop_w // 2)))
        y1 = max(0, min(frame_h - crop_h, center_y_px - (crop_h // 2)))
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        final_clip = subclip.crop(x1=x1, y1=y1, x2=x2, y2=y2)
        final_clip.write_videofile(
            final_output_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=os.path.join(output_dir, "temp-audio.m4a"),
            remove_temp=True,
            logger=None,
        )
        final_clip.close()
        return final_output_path
    finally:
        if subclip is not None:
            subclip.close()
        if clip is not None:
            clip.close()
