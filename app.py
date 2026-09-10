import sys
import types
import soundfile as sf
import torch

# Patch 1: PyTorch 2.6+ weights_only default fix
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Patch 2: torchaudio.backend compatibility mapping
import torchaudio
if 'torchaudio.backend' not in sys.modules:
    b = types.ModuleType('backend')
    c = types.ModuleType('common')
    c.AudioMetaData = getattr(torchaudio, 'AudioMetaData', None)
    b.common = c
    sys.modules['torchaudio.backend'] = b
    sys.modules['torchaudio.backend.common'] = c

# Patch 3: torchaudio.info fallback for newer torchaudio versions
if not hasattr(torchaudio, 'info'):
    class FakeAudioMetaData:
        def __init__(self, sample_rate, num_frames, num_channels):
            self.sample_rate = sample_rate
            self.num_frames = num_frames
            self.num_channels = num_channels
            self.bits_per_sample = 16
            self.encoding = 'PCM_S'
    def _patched_info(filepath, **kwargs):
        info = sf.info(filepath)
        return FakeAudioMetaData(info.samplerate, info.frames, info.channels)
# Patch 4: Windows EventLoopPolicy fix to eliminate WinError 10054 asyncio socket reset
import asyncio
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import gradio as gr
import os
# os.system("python -m unidic download")
from pytubefix import YouTube
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
import whisper
from moviepy.editor import *
from pydub import AudioSegment
from df.enhance import enhance, init_df, load_audio, save_audio
import translators as ts
from melo.api import TTS
from concurrent.futures import ThreadPoolExecutor
import ffmpeg
import nltk
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    try:
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    except Exception:
        pass

SUBTITLE_BG_COLORS = {
    "Black (Opaque) / 黑色 (不透明)": "&H00000000",
    "Black (Semi-Transparent) / 黑色 (半透明)": "&H80000000",
    "Dark Gray / 深灰色 (不透明)": "&H00333333",
    "White / 白色 (不透明)": "&H00FFFFFF",
    "Navy Blue / 深蓝色 (不透明)": "&H00660000",
}

def process_upload(video_file, language_choice, enable_sub_bg=True, sub_bg_color="Black (Opaque) / 黑色 (不透明)"):
    if language_choice == None:
        return None, "Language not selected."
    elif video_file == None:
        return None, "Video not uploaded."
    else:
        video_path = video_file.name if hasattr(video_file, 'name') and video_file.name else video_file
        return process_video(video_path, language_choice, enable_sub_bg, sub_bg_color)

def process_youtube(youtube_url, language_choice, enable_sub_bg=True, sub_bg_color="Black (Opaque) / 黑色 (不透明)"):
    if language_choice is None:
        return None, "Language not selected."
    elif youtube_url is None:
        return None, "YouTube URL not entered."
    
    video_file = "original.mp4"
    if os.path.exists(video_file):
        try:
            os.remove(video_file)
        except Exception:
            pass

    download_success = False

    # Try pytubefix first
    try:
        yt = YouTube(youtube_url)
        stream = (
            yt.streams.filter(progressive=True, file_extension='mp4').first()
            or yt.streams.get_highest_resolution()
            or yt.streams.filter(file_extension='mp4').first()
            or yt.streams.first()
        )
        if stream:
            stream.download(filename=video_file)
            download_success = os.path.exists(video_file)
    except Exception as e:
        print(f"pytubefix download failed: {e}, trying yt_dlp...")

    # Fallback to yt_dlp Python library
    if not download_success or not os.path.exists(video_file):
        try:
            import yt_dlp
            ydl_opts = {
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                'outtmpl': video_file,
                'quiet': True,
                'no_warnings': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
            download_success = os.path.exists(video_file)
        except Exception as e:
            print(f"yt_dlp python module error: {e}")

    # Fallback to yt-dlp CLI command
    if not download_success or not os.path.exists(video_file):
        try:
            import subprocess
            cmd = ["yt-dlp", "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best", "-o", video_file, youtube_url]
            subprocess.run(cmd, check=True)
            download_success = os.path.exists(video_file)
        except Exception as e:
            print(f"yt-dlp CLI fallback error: {e}")

    if not download_success or not os.path.exists(video_file):
        return None, "Failed to download YouTube video. Please ensure yt-dlp is installed (`pip install -U yt-dlp pytubefix`) or upload the video file directly."

    return process_video(video_file, language_choice, enable_sub_bg, sub_bg_color)

def process_video(video_file, language_choice, enable_sub_bg=True, sub_bg_color="Black (Opaque) / 黑色 (不透明)"):
    # Initialize paths and devices
    ckpt_converter = 'checkpoints_v2/converter'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_dir = 'outputs_v2'
    os.makedirs(output_dir, exist_ok=True)

    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

    # Process the reference video
    reference_video = VideoFileClip(video_file)
    reference_audio = os.path.join(output_dir, "reference_audio.wav")

    if reference_video.audio is not None:
        reference_video.audio.write_audiofile(reference_audio)
    else:
        # Try extracting audio using ffmpeg directly
        try:
            (
                ffmpeg
                .input(video_file)
                .output(reference_audio, acodec='pcm_s16le', ac=1, ar='48000')
                .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error:
            return None, "The selected video file does not contain any audio track to translate."

    if not os.path.exists(reference_audio) or os.path.getsize(reference_audio) == 0:
        return None, "Failed to extract audio track from video."
    audio = AudioSegment.from_file(reference_audio)
    resampled_audio = audio.set_frame_rate(48000)
    resampled_audio.export(reference_audio, format="wav")

    # Enhance the audio
    model, df_state, _ = init_df()
    audio, _ = load_audio(reference_audio, sr=df_state.sr())
    enhanced = enhance(model, df_state, audio)
    save_audio(reference_audio, enhanced, df_state.sr())
    audio_clip = AudioFileClip(reference_audio)

    src_path = os.path.join(output_dir, "tmp.wav")

    # Speed is adjustable
    speed = 1.0

    # Transcribe the original audio with timestamps using upgraded Whisper model
    sttmodel = whisper.load_model("medium")
    sttresult = sttmodel.transcribe(reference_audio, initial_prompt="以下是简体普通话内容：", verbose=True)

    # Print the original transcription
    print(sttresult["text"])
    print(sttresult["language"])

    # Get the segments with start and end times
    segments = sttresult['segments']

    if sttresult["language"] == language_choice[0:2]:
        print("Chosen language is the same as the video's original language. Only adding subtitles.")
        segments = sttresult['segments']

        # Generate subtitles file in SRT format
        srt_path = os.path.join(output_dir, 'subtitles.srt')
        with open(srt_path, 'w', encoding='utf-8') as srt_file:
            for i, segment in enumerate(segments):
                start = segment['start']
                end = segment['end']
                text = segment['text']

                start_hours, start_minutes = divmod(int(start), 3600)
                start_minutes, start_seconds = divmod(start_minutes, 60)
                start_milliseconds = int((start * 1000) % 1000)

                end_hours, end_minutes = divmod(int(end), 3600)
                end_minutes, end_seconds = divmod(end_minutes, 60)
                end_milliseconds = int((end * 1000) % 1000)

                srt_file.write(f"{i+1}\n")
                srt_file.write(f"{start_hours:02}:{start_minutes:02}:{start_seconds:02},{start_milliseconds:03} --> "
                               f"{end_hours:02}:{end_minutes:02}:{end_seconds:02},{end_milliseconds:03}\n")
                srt_file.write(f"{text}\n\n")

        # Add subtitles to the video
        final_video_with_subs_path = os.path.join(output_dir, f'final_video_with_subs.mp4')
        srt_path_ffmpeg = srt_path.replace('\\', '/')
        try:
            (
                ffmpeg
                .input(video_file)
                .output(final_video_with_subs_path, vf=f"subtitles='{srt_path_ffmpeg}':force_style='FontSize=18'")
                .run(overwrite_output=True)
            )
        except ffmpeg.Error as e:
            print('ffmpeg error:', e)
            if getattr(e, 'stderr', None):
                print(e.stderr.decode('utf-8', errors='ignore'))

        print(f"Final video with subtitles saved to: {final_video_with_subs_path}")
        return final_video_with_subs_path, "Video language and language selection are the same, audio not changed."
    else:        
        # Choose the target language for translation
        language = 'EN_NEWEST'
        match language_choice[0:2]:
            case 'en':
                language = 'EN_NEWEST'
            case 'es':
                language = 'ES'
            case 'fr':
                language = 'FR'
            case 'zh':
                language = 'ZH'
            case 'ja':
                language = 'JP'
            case 'ko':
                language = 'KR'
            case _:
                language = 'EN_NEWEST'
    
        # Translate the transcription segment by segment
        def translate_segment(segment):
            return segment["start"], segment["end"], ts.translate_text(query_text=segment["text"], translator="google", to_language=language_choice)
    
        # Batch translation to reduce memory load
        batch_size = 2
        translation_segments = []
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            with ThreadPoolExecutor(max_workers=5) as executor:
                batch_translations = list(executor.map(translate_segment, batch))
            translation_segments.extend(batch_translations)
    
        # Generate the translated audio for each segment
        model = TTS(language=language, device=device)
        speaker_ids = model.hps.data.spk2id
    
        def generate_segment_audio_batch(translation_batch, speaker_id):
            segment_files = []
            total_duration = audio_clip.duration
            for segment in translation_batch:
                start, end, translated_text = segment
                start = max(0.0, round(start, 2))
                end = round(end, 2)

                # Clamp start and end safely within total clip duration
                if start >= total_duration:
                    start = max(0.0, total_duration - 0.5)
                end = min(end, total_duration)

                if end <= start:
                    end = min(start + 0.5, total_duration)

                segment_path = os.path.join(output_dir, f'segment_{start}_{end}.wav')
                model.tts_to_file(translated_text, speaker_id, segment_path, speed=speed)

                ref_speaker_file = os.path.join(output_dir, f'reference_speaker_{start}_{end}.wav')
                try:
                    if start < total_duration and (end - start) > 0.05:
                        reference_speaker = AudioFileClip.subclip(audio_clip, start, end)
                        reference_speaker.write_audiofile(ref_speaker_file)
                        target_se, audio_name = se_extractor.get_se(ref_speaker_file, tone_color_converter, vad=False)
                    else:
                        target_se, audio_name = se_extractor.get_se(reference_audio, tone_color_converter, vad=False)
                except Exception:
                    target_se, audio_name = se_extractor.get_se(reference_audio, tone_color_converter, vad=False)

                # Run the tone color converter
                encode_message = "@MyShell"
                tone_color_converter.convert(
                    audio_src_path=segment_path,
                    src_se=source_se,
                    tgt_se=target_se,
                    output_path=segment_path,
                    message=encode_message
                )
                
                segment_files.append((segment_path, start, end, translated_text))
            return segment_files
    
        for speaker_key in speaker_ids.keys():
            speaker_id = speaker_ids[speaker_key]
            speaker_key = speaker_key.lower().replace('_', '-')
    
            source_se = torch.load(f'checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=device)
    
            segment_files = []
            subtitle_entries = []
            for i in range(0, len(translation_segments), batch_size):
                batch = translation_segments[i:i + batch_size]
                with ThreadPoolExecutor(max_workers=5) as executor:
                    batch_segment_files = list(executor.map(generate_segment_audio_batch, [batch] * len(speaker_ids), [speaker_id] * len(batch)))
                    batch_segment_files = [item for sublist in batch_segment_files for item in sublist]  # Flatten the list
    
                for segment_file, start, end, translated_text in batch_segment_files:                    
                    segment_files.append((segment_file, start, end, translated_text))
    
            # Combine the audio segments
            combined_audio = AudioSegment.empty()
            video_segments = []
            previous_end = 0
            subtitle_counter = 1
            for segment_file, start, end, translated_text in segment_files:
                segment_audio = AudioSegment.from_file(segment_file)
                combined_audio += segment_audio
                
                # Calculate the duration of the audio segment
                audio_duration = len(segment_audio) / 1000.0
    
                # Add the subtitle entry for this segment
                subtitle_entries.append((subtitle_counter, previous_end, previous_end + audio_duration, translated_text))
                subtitle_counter += 1
    
                # Safe PTS scale calculation
                seg_duration = max(0.1, end - start)
                pts_scale = seg_duration / audio_duration if audio_duration > 0 else 1.0

                # Get the corresponding video segment and adjust its speed to match the audio duration
                video_segment = (
                    ffmpeg
                    .input(reference_video.filename, ss=start, to=end)
                    .filter('setpts', f'PTS / {pts_scale}')
                )
                video_segments.append((video_segment, ffmpeg.input(segment_file)))
                previous_end += audio_duration
    
            save_path = os.path.join(output_dir, f'output_v2_{speaker_key}.wav')
            combined_audio.export(save_path, format="wav")
    
            # Combine video and audio segments using ffmpeg
            video_and_audio_files = [item for sublist in video_segments for item in sublist]
            joined = (
                ffmpeg
                .concat(*video_and_audio_files, v=1, a=1)
                .node
            )
    
            final_video_path = os.path.join(output_dir, f'final_video_{speaker_key}.mp4')
            try:
                (
                    ffmpeg
                    .output(joined[0], joined[1], final_video_path, vcodec='libx264', acodec='aac')
                    .run(overwrite_output=True)
                )
            except ffmpeg.Error as e:
                print('ffmpeg error:', e)
                if getattr(e, 'stderr', None):
                    print(e.stderr.decode('utf-8', errors='ignore'))
    
            print(f"Final video without subtitles saved to: {final_video_path}")
    
            # Generate subtitles file in SRT format
            srt_path = os.path.join(output_dir, 'subtitles.srt')
            with open(srt_path, 'w', encoding='utf-8') as srt_file:
                for entry in subtitle_entries:
                    index, start, end, text = entry
                    start_hours, start_minutes = divmod(int(start), 3600)
                    start_minutes, start_seconds = divmod(start_minutes, 60)
                    start_milliseconds = int((start * 1000) % 1000)
    
                    end_hours, end_minutes = divmod(int(end), 3600)
                    end_minutes, end_seconds = divmod(end_minutes, 60)
                    end_milliseconds = int((end * 1000) % 1000)
    
                    srt_file.write(f"{index}\n")
                    srt_file.write(f"{start_hours:02}:{start_minutes:02}:{start_seconds:02},{start_milliseconds:03} --> "
                                   f"{end_hours:02}:{end_minutes:02}:{end_seconds:02},{end_milliseconds:03}\n")
                    srt_file.write(f"{text}\n\n")
    
            # Add subtitles to the video
            final_video_with_subs_path = os.path.join(output_dir, f'final_video_with_subs_{speaker_key}.mp4')
            srt_path_ffmpeg = srt_path.replace('\\', '/')

            if enable_sub_bg:
                bg_color_code = SUBTITLE_BG_COLORS.get(sub_bg_color, "&H00000000")
                force_style = f"BorderStyle=3,BackColour={bg_color_code},OutlineColour={bg_color_code},Outline=4,FontSize=18,PrimaryColour=&H00FFFFFF"
            else:
                force_style = "FontSize=18"

            try:
                (
                    ffmpeg
                    .input(final_video_path)
                    .output(final_video_with_subs_path, vf=f"subtitles='{srt_path_ffmpeg}':force_style='{force_style}'")
                    .run(overwrite_output=True)
                )
            except ffmpeg.Error as e:
                print('ffmpeg error:', e)
                if getattr(e, 'stderr', None):
                    print(e.stderr.decode('utf-8', errors='ignore'))
    
            print(f"Final video with subtitles saved to: {final_video_with_subs_path}")
    
            return final_video_with_subs_path, "Video successfully translated."

# Gradio Interface (Restricted to languages supported by MeloTTS)
language_choices = ["en", "zh-cn", "es", "fr", "ja", "ko"]
bg_color_choices = list(SUBTITLE_BG_COLORS.keys())

uploaded_translator = gr.Interface(
    fn=process_upload,
    inputs=[
        gr.File(label="Upload a video file (.mp4, .mkv, .avi, .mov, .flv, etc.)", file_types=['video', '.mp4', '.mkv', '.avi', '.mov', '.flv', '.webm']),
        gr.Dropdown(choices=language_choices, value="zh-cn", label="Choose Language for Translation (Expressed in ISO 639-1 code)"),
        gr.Checkbox(label="开启字幕背景块 (遮挡原视频字幕) / Enable Subtitle Background Box (Mask Original Subtitles)", value=True),
        gr.Dropdown(choices=bg_color_choices, value="Black (Opaque) / 黑色 (不透明)", label="字幕背景块颜色 / Subtitle Background Color")
    ],
    outputs=[
        gr.Video(label="Translated Video", format='mp4'),
        gr.Textbox(show_label=False)
    ],
    title="Video Translation and Voice Cloning",
    description="Upload a video, choose a language to translate the audio, and download the processed video with translated audio."
)

youtube_translator = gr.Interface(
    fn=process_youtube,
    inputs=[
        gr.Textbox(label="Enter a YouTube video URL"),
        gr.Dropdown(choices=language_choices, value="zh-cn", label="Choose Language for Translation (Expressed in ISO 639-1 code)"),
        gr.Checkbox(label="开启字幕背景块 (遮挡原视频字幕) / Enable Subtitle Background Box (Mask Original Subtitles)", value=True),
        gr.Dropdown(choices=bg_color_choices, value="Black (Opaque) / 黑色 (不透明)", label="字幕背景块颜色 / Subtitle Background Color")
    ],
    outputs=[
        gr.Video(label="Translated Video", format='mp4'),
        gr.Textbox(show_label=False)
    ],
    title="Video Translation and Voice Cloning",
    description="Upload a video, choose a language to translate the audio, and download the processed video with translated audio."
)

gr.TabbedInterface([uploaded_translator, youtube_translator], ["Upload video from device", "YouTube URL"]).launch(share=True)
