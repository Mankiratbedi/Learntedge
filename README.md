# LEARNTEDGE
**Transforming Long-Form Mentorship into Viral Shorts**

### 🎥 [CLICK HERE FOR LIVE DEMO](https://drive.google.com/file/d/1Yp0nLY3inKAgEKxi1ahqArbE-I5oPnXJ/view?usp=sharing)

### ## Overview
LearntEdge uses **Gemini 3 Flash** and **Multimodal Deep Learning** to identify "Emotional Peaks" and "Golden Nuggets" in educational videos, automatically cropping them to a 9:16 vertical format for social media.

### ## Tech Stack
* **AI Brain:** Gemini 3 Flash (via `google-generativeai`)
* **Transcription:** OpenAI Whisper
* **Video Engine:** MoviePy 2.0
* **Frontend:** Streamlit

## ⚙️ How it Works
Ingestion: The user uploads an MP4 or provides a transcript.

Transcription: Whisper extracts precise timestamps for every word.

Intelligence: Gemini 3 Flash identifies the "Golden Nuggets" (30–60s segments) that have the highest viral potential.

Transformation: MoviePy performs a mathematical center-crop to 9:16 and exports the final high-definition clips.
