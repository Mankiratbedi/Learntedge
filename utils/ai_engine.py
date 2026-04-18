import os
import json
from typing import List, Dict

import google.generativeai as genai

from dotenv import load_dotenv

load_dotenv()


def extract_viral_moments(transcript_with_timestamps: str) -> List[Dict]:
    """
    Analyze a timestamped video transcript and return 3 viral moments.

    Returns:
        [
          {
            "start_time": <int seconds>,
            "end_time": <int seconds>,
            "hook_headline": <str>,
            "reason": <str>
          },
          ...
        ]
    """

    # Basic input validation
    if not transcript_with_timestamps or len(transcript_with_timestamps.strip()) < 300:
        raise ValueError(
            "Transcript is too short. Provide a longer timestamped transcript."
        )

    # API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing GEMINI_API_KEY in environment variables.")

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = f"""
You are an expert social media editor.

Task:
Given this timestamped transcript, find exactly 3 viral moments ("golden nuggets") based on:
- profound wisdom
- high-energy / emotional sentiment
- scroll-stopping potential

Output rules:
1) Return ONLY valid JSON (no markdown, no extra text).
2) Output must be a JSON list of exactly 3 objects.
3) Each object must include:
   - start_time (integer seconds)
   - end_time (integer seconds, always start_time + 60)
   - hook_headline (catchy, short, stop-the-scroll style)
   - reason (brief explanation of why it's viral)
4) If transcript timestamps are in mm:ss or hh:mm:ss, convert to total seconds.

Transcript:
{transcript_with_timestamps}
"""

        response = model.generate_content(prompt)
        raw = response.text.strip()

        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.replace("json", "", 1).strip()

        data = json.loads(raw)

        # Validate final shape
        if not isinstance(data, list) or len(data) != 3:
            raise ValueError("Model did not return exactly 3 moments.")

        for item in data:
            if not all(k in item for k in ("start_time", "end_time", "hook_headline", "reason")):
                raise ValueError("Missing required fields in one or more objects.")

            # enforce integer + 60s rule
            start = int(item["start_time"])
            item["start_time"] = start
            item["end_time"] = start + 60

        return data

    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse model output as JSON: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Gemini analysis failed: {e}") from e