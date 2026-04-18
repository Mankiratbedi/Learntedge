from moviepy import VideoFileClip
import os

def process_video_clip(input_path, output_path, start_time, end_time):
    """
    Emergency Fallback: Crops video to 9:16 vertical using center-alignment.
    Bypasses MediaPipe to ensure the app works for the deadline.
    """
    try:
        # Load the video
        clip = VideoFileClip(input_path).subclipped(start_time, end_time)
        
        # Calculate 9:16 Crop
        w, h = clip.size
        target_ratio = 9/16
        target_w = h * target_ratio
        
        # Center the crop
        x_center = w / 2
        x1 = max(0, x_center - target_w / 2)
        x2 = min(w, x_center + target_w / 2)
        
        # Crop and Save
        final_clip = clip.cropped(x1=x1, y1=0, x2=x2, y2=h)
        
        output_path = os.path.join("output", output_path)
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        
        return output_path
    except Exception as e:
        raise RuntimeError(f"Video processing failed: {e}")