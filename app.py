import os
from dotenv import load_dotenv
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

from utils.ai_engine import extract_viral_moments
from utils.video_processor import process_video_clip

load_dotenv()


def _init_state() -> None:
    defaults = {
        "uploaded_video_path": None,
        "uploaded_video_name": None,
        "transcript_text": "",
        "viral_moments": [],
        "processed_clips": [],
        "last_error": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _save_uploaded_video(uploaded_file) -> str:
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)
    file_path = temp_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(file_path.resolve())


def _format_transcript_with_timestamps(segments: List[Dict]) -> str:
    lines: List[str] = []
    for seg in segments:
        start = int(seg.get("start", 0))
        mm = start // 60
        ss = start % 60
        text = str(seg.get("text", "")).strip()
        if text:
            lines.append(f"[{mm:02d}:{ss:02d}] {text}")
    return "\n".join(lines)


def _generate_transcript(video_path: str, model_name: str) -> str:
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError(
            "openai-whisper is not installed. Install it with: pip install openai-whisper"
        ) from exc

    model = whisper.load_model(model_name)
    result = model.transcribe(video_path)
    segments = result.get("segments", [])
    if not segments:
        raise ValueError("Whisper could not generate transcript segments.")
    return _format_transcript_with_timestamps(segments)


def _render_results_grid(clips: List[Dict]) -> None:
    if not clips:
        return

    st.subheader("Generated Viral Shorts")
    cols = st.columns(3)
    for idx, clip_data in enumerate(clips):
        with cols[idx % 3]:
            clip_path = clip_data["clip_path"]
            with open(clip_path, "rb") as f:
                clip_bytes = f.read()
            st.video(clip_bytes)
            st.markdown(f"**{clip_data['hook_headline']}**")
            st.caption(clip_data.get("reason", ""))
            st.download_button(
                label="Download Clip",
                data=clip_bytes,
                file_name=Path(clip_path).name,
                mime="video/mp4",
                key=f"download_{idx}_{Path(clip_path).name}",
            )


def main() -> None:
    st.set_page_config(page_title="LearntEdge - Viral Shorts", layout="wide")
    _init_state()

    st.title("LearntEdge Viral Shorts Generator")

    with st.sidebar:
        st.header("Settings")
        whisper_model = st.selectbox(
            "Whisper Model",
            options=["base", "small", "medium"],
            index=0,
            help="Smaller models are faster; larger models can be more accurate.",
        )
        st.markdown("Output clips are saved in the `output/` folder.")

    st.subheader("1) Upload Video")
    uploaded_file = st.file_uploader(
        "Upload an MP4 video", type=["mp4"], accept_multiple_files=False
    )

    if uploaded_file is not None:
        current_name = uploaded_file.name
        if st.session_state.uploaded_video_name != current_name:
            saved_path = _save_uploaded_video(uploaded_file)
            st.session_state.uploaded_video_path = saved_path
            st.session_state.uploaded_video_name = current_name
            st.session_state.processed_clips = []
            st.session_state.viral_moments = []
        st.success(f"Video ready: {st.session_state.uploaded_video_name}")

    st.subheader("2) Transcript")
    transcript_text = st.text_area(
        "Paste transcript with timestamps (optional if using Whisper)",
        value=st.session_state.transcript_text,
        height=220,
        placeholder="[00:12] Your transcript line here...",
    )
    st.session_state.transcript_text = transcript_text

    if st.button("Generate Transcript"):
        if not st.session_state.uploaded_video_path:
            st.error("Upload an MP4 file first.")
        elif st.session_state.transcript_text.strip():
            st.info("Transcript already exists in the text area.")
        else:
            with st.spinner("Generating transcript with Whisper..."):
                try:
                    transcript = _generate_transcript(
                        st.session_state.uploaded_video_path, whisper_model
                    )
                    st.session_state.transcript_text = transcript
                    st.success("Transcript generated.")
                except Exception as exc:
                    st.error(f"Failed to generate transcript: {exc}")

    if st.session_state.transcript_text.strip():
        st.text_area(
            "Transcript Preview",
            value=st.session_state.transcript_text,
            height=220,
            disabled=True,
        )

    st.subheader("3) Generate Viral Shorts")
    if st.button("Generate Viral Shorts"):
        if not st.session_state.uploaded_video_path:
            st.error("Please upload an MP4 video first.")
        elif not st.session_state.transcript_text.strip():
            st.error("Transcript is required. Paste one or click Generate Transcript.")
        else:
            try:
                moments = extract_viral_moments(st.session_state.transcript_text)
                st.session_state.viral_moments = moments

                progress = st.progress(0)
                clips: List[Dict] = []

                for idx, moment in enumerate(moments):
                    start_time = int(moment["start_time"])
                    end_time = int(moment["end_time"])
                    clip_filename = f"viral_short_{idx + 1}.mp4"
                    clip_path = process_video_clip(
                        input_path=st.session_state.uploaded_video_path,
                        output_path=clip_filename,
                        start_time=start_time,
                        end_time=end_time,
                    )
                    clips.append(
                        {
                            "clip_path": clip_path,
                            "hook_headline": moment.get(
                                "hook_headline", f"Viral Clip {idx + 1}"
                            ),
                            "reason": moment.get("reason", ""),
                            "start_time": start_time,
                            "end_time": end_time,
                        }
                    )
                    progress.progress((idx + 1) / 3)

                st.session_state.processed_clips = clips
                st.success("All viral shorts generated successfully.")
            except Exception as exc:
                st.error(f"Failed to generate viral shorts: {exc}")

    _render_results_grid(st.session_state.processed_clips)


if __name__ == "__main__":
    main()
