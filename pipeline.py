"""
Core pipeline for turning a YouTube URL into captioned highlight clips.

Steps:
1. download_video()      - pulls the video with yt-dlp
2. transcribe()          - runs faster-whisper, returns word/segment level timestamps
3. score_by_audio_energy()  - finds loud/energetic moments (laughter, shouting, hype)
4. pick_highlights_by_energy() - picks the best non-overlapping clip windows
5. cut_and_caption()      - uses ffmpeg to cut each window and burn in captions
6. generate_caption() / generate_thumbnail() - post text + thumbnail image
"""

import os
import json
import subprocess
import wave
import contextlib
import audioop
import math
from dataclasses import dataclass, asdict
from typing import List

import yt_dlp
from faster_whisper import WhisperModel

OUTPUT_DIR = "outputs"
DOWNLOAD_DIR = "downloads"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Clip:
    start: float
    end: float
    reason: str
    score: float


# ---------- 1. Download ----------

def _get_cookiefile():
    """If a YOUTUBE_COOKIES env var is set (paste of a cookies.txt export),
    write it to a temp file so yt-dlp can use it to look like a logged-in
    browser instead of anonymous/bot traffic -- needed on cloud hosts like
    Render, which YouTube often blocks otherwise."""
    cookies_content = os.environ.get("YOUTUBE_COOKIES")
    if not cookies_content:
        return None
    cookie_path = os.path.join(DOWNLOAD_DIR, "cookies.txt")
    with open(cookie_path, "w") as f:
        f.write(cookies_content)
    return cookie_path


def download_video(url: str, job_id: str):
    out_path = os.path.join(DOWNLOAD_DIR, f"{job_id}.mp4")
    ydl_opts = {
        "format": "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": out_path,
        "merge_output_format": "mp4",
        "quiet": True,
        "noplaylist": True,
        # The "android" client often sidesteps YouTube's "sign in to confirm
        # you're not a bot" check that hits cloud-hosted IPs, even without
        # cookies. We combine it with cookies (if provided) for best odds.
        "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
        "user_agent": "com.google.android.youtube/19.09.37 (Linux; U; Android 14) gzip",
    }
    cookiefile = _get_cookiefile()
    if cookiefile:
        ydl_opts["cookiefile"] = cookiefile
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
    title = info.get("title", "")
    return out_path, title


# ---------- 2. Transcribe ----------

_model = None

def get_model():
    global _model
    if _model is None:
        # "base" fits comfortably in low-memory hosting (e.g. Render free tier).
        # If you're running this locally with plenty of RAM, change to "small"
        # or "medium" for noticeably better transcription accuracy.
        model_size = os.environ.get("WHISPER_MODEL", "base")
        _model = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _model


def transcribe(video_path: str) -> List[Segment]:
    model = get_model()
    segments, _info = model.transcribe(video_path, word_timestamps=True)
    result = []
    for seg in segments:
        result.append(Segment(start=seg.start, end=seg.end, text=seg.text.strip()))
    return result


def segments_to_srt(segments: List[Segment], offset: float = 0.0) -> str:
    """Build an SRT string for a slice of segments, shifting timestamps so the
    clip's captions start at 0."""
    def fmt(t):
        if t < 0:
            t = 0
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int((t - math.floor(t)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines = []
    for i, seg in enumerate(segments, 1):
        start = seg.start - offset
        end = seg.end - offset
        lines.append(str(i))
        lines.append(f"{fmt(start)} --> {fmt(end)}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)


# ---------- 3. Audio energy scoring ----------

def extract_wav(video_path: str, job_id: str) -> str:
    wav_path = os.path.join(DOWNLOAD_DIR, f"{job_id}.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000", wav_path],
        check=True, capture_output=True,
    )
    return wav_path


def score_by_audio_energy(wav_path: str, window_sec: float = 3.0) -> List[tuple]:
    """Returns list of (start_time, energy_score) for each window across the file.
    Loud/energetic windows (laughter, shouting, music stings) score higher."""
    with contextlib.closing(wave.open(wav_path, "rb")) as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    frames_per_window = int(framerate * window_sec)
    bytes_per_frame = sampwidth * n_channels
    window_bytes = frames_per_window * bytes_per_frame

    scores = []
    for i in range(0, len(raw), window_bytes):
        chunk = raw[i:i + window_bytes]
        if len(chunk) < bytes_per_frame:
            break
        rms = audioop.rms(chunk, sampwidth)
        t = i / bytes_per_frame / framerate
        scores.append((t, rms))

    if not scores:
        return []
    max_rms = max(s for _, s in scores) or 1
    return [(t, rms / max_rms) for t, rms in scores]


# ---------- 4. Highlight selection (audio energy only, no API key needed) ----------

def pick_highlights_by_energy(energy_scores: List[tuple], segments: List[Segment],
                                num_clips: int, clip_len: int, step: float = 5.0) -> List[Clip]:
    """Slide a clip_len-second window across the whole video, score each position
    by its average audio energy, and greedily pick the highest-scoring
    non-overlapping windows. Nudges each window to start at the nearest
    sentence boundary so clips don't begin mid-word."""
    if not energy_scores:
        return []

    total_duration = energy_scores[-1][0]
    candidates = []
    t = 0.0
    while t + clip_len <= total_duration:
        pts = [s for time, s in energy_scores if t <= time <= t + clip_len]
        avg_score = sum(pts) / len(pts) if pts else 0.0
        candidates.append((t, avg_score))
        t += step

    candidates.sort(key=lambda c: c[1], reverse=True)

    chosen = []
    for start, score in candidates:
        end = start + clip_len
        if any(not (end <= c_start or start >= c_end) for c_start, c_end in chosen):
            continue  # overlaps an already-chosen clip
        chosen.append((start, end))
        if len(chosen) >= num_clips:
            break

    def snap_to_sentence_start(t):
        best = t
        best_diff = 999999
        for s in segments:
            diff = abs(s.start - t)
            if diff < best_diff and diff < 4.0:
                best_diff = diff
                best = s.start
        return best

    clips = []
    for start, end in chosen:
        snapped_start = snap_to_sentence_start(start)
        clips.append(Clip(start=snapped_start, end=snapped_start + clip_len,
                           reason="High-energy moment (loud/animated audio)", score=1.0))

    return sorted(clips, key=lambda c: c.start)


# ---------- 5. Cut + caption ----------

def cut_and_caption(video_path: str, segments: List[Segment], clip: Clip, job_id: str, idx: int,
                     vertical: bool = True) -> str:
    clip_segments = [s for s in segments if s.end > clip.start and s.start < clip.end]
    srt_content = segments_to_srt(clip_segments, offset=clip.start)
    srt_path = os.path.join(OUTPUT_DIR, f"{job_id}_clip{idx}.srt")
    with open(srt_path, "w") as f:
        f.write(srt_content)

    out_path = os.path.join(OUTPUT_DIR, f"{job_id}_clip{idx}.mp4")
    duration = clip.end - clip.start

    tmp_cut = os.path.join(OUTPUT_DIR, f"{job_id}_clip{idx}_raw.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(clip.start), "-i", video_path, "-t", str(duration),
         "-c:v", "libx264", "-c:a", "aac", tmp_cut],
        check=True, capture_output=True,
    )

    caption_style = "FontSize=26,PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,BorderStyle=3,Alignment=2,MarginV=90"

    if vertical:
        vf = (
            "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,gblur=sigma=25[bg];"
            "[0:v]scale=1080:-2[fg];"
            "[bg][fg]overlay=(W-w)/2:(H-h)/2,"
            f"subtitles={srt_path}:force_style='{caption_style}'"
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_cut, "-filter_complex", vf,
             "-c:v", "libx264", "-c:a", "aac", out_path],
            check=True, capture_output=True,
        )
    else:
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_cut,
             "-vf", f"subtitles={srt_path}:force_style='{caption_style}'",
             "-c:a", "copy", out_path],
            check=True, capture_output=True,
        )
    os.remove(tmp_cut)
    return out_path


# ---------- 6. Caption text + thumbnail ----------

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "to", "of", "in", "on", "for", "with", "this", "that", "it",
    "as", "at", "by", "from", "so", "just", "like", "you", "your", "i",
    "we", "he", "she", "they", "them", "his", "her", "their", "have", "has",
    "had", "do", "does", "did", "not", "no", "yes", "if", "then", "than",
    "get", "got", "going", "gonna", "know", "think", "really", "very",
}


def generate_caption(clip_segments: List[Segment], video_title: str) -> str:
    """Builds a short social-caption (the text that goes under the post) from
    the clip's own transcript, plus a few relevant hashtags -- no AI needed."""
    text = " ".join(s.text for s in clip_segments).strip()
    if not text:
        text = video_title or "Check this out"

    hook = text.split(".")[0].strip()
    if len(hook) > 120:
        hook = hook[:117].rsplit(" ", 1)[0] + "..."
    if not hook:
        hook = text[:100]

    punctuation = '.,!?"\''
    words = [w.strip(punctuation).lower() for w in text.split()]
    keywords = []
    for w in words:
        if len(w) > 3 and w not in STOPWORDS and w.isalpha() and w not in keywords:
            keywords.append(w)
        if len(keywords) >= 4:
            break

    hashtags = ["#shorts", "#fyp", "#viral"] + [f"#{w}" for w in keywords]
    caption = f"{hook}\n\n{' '.join(hashtags)}"
    return caption


def generate_thumbnail(clip_path: str, job_id: str, idx: int) -> str:
    """Grabs a frame from partway into the clip to use as a thumbnail."""
    thumb_path = os.path.join(OUTPUT_DIR, f"{job_id}_clip{idx}_thumb.jpg")
    subprocess.run(
        ["ffmpeg", "-y", "-i", clip_path, "-ss", "00:00:01", "-frames:v", "1",
         "-q:v", "2", thumb_path],
        check=True, capture_output=True,
    )
    return thumb_path


# ---------- Orchestration ----------

def run_pipeline(url: str, job_id: str, num_clips: int = 5, clip_len: int = 45, progress_cb=None):
    def report(msg):
        if progress_cb:
            progress_cb(msg)

    report("Downloading video...")
    video_path, video_title = download_video(url, job_id)

    report("Transcribing audio...")
    segments = transcribe(video_path)

    report("Analyzing audio energy...")
    wav_path = extract_wav(video_path, job_id)
    energy_scores = score_by_audio_energy(wav_path)

    report("Picking highlight moments from audio energy...")
    final_clips = pick_highlights_by_energy(energy_scores, segments, num_clips=num_clips, clip_len=clip_len)

    results = []
    for i, clip in enumerate(final_clips, 1):
        report(f"Cutting & captioning video clip {i}/{len(final_clips)}...")
        out_path = cut_and_caption(video_path, segments, clip, job_id, i)

        report(f"Generating thumbnail {i}/{len(final_clips)}...")
        thumb_path = generate_thumbnail(out_path, job_id, i)

        clip_segments = [s for s in segments if s.end > clip.start and s.start < clip.end]
        caption_text = generate_caption(clip_segments, video_title)

        results.append({
            "file": os.path.basename(out_path),
            "thumbnail": os.path.basename(thumb_path),
            "caption": caption_text,
            "start": clip.start,
            "end": clip.end,
            "reason": clip.reason,
            "score": round(clip.score, 3),
        })

    report("Done.")
    return results
