import os
import atexit
from flask import Flask, request, jsonify, render_template, Response
from moviepy import VideoFileClip
import speech_recognition as sr
import instaloader
import time  # For simulating progress (if needed)
import threading 

app = Flask(__name__)

# Temporary directory to store downloaded files
TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)

def download_instagram_reel(link):
    """Download Instagram Reel and return the path to the video file."""
    try:
        loader = instaloader.Instaloader()
        loader.download_video = True
        loader.dirname_pattern = TEMP_DIR  # Store downloaded files in TEMP_DIR

        # Extract shortcode from the link
        shortcode = link.split("/")[-2]
        print(f"Shortcode extracted from link: {shortcode}")

        # Download the post
        post = instaloader.Post.from_shortcode(loader.context, shortcode)
        loader.download_post(post, target=TEMP_DIR)

        # Get all files in TEMP_DIR
        downloaded_files = os.listdir(TEMP_DIR)

        # Debugging log: print all downloaded files
        print(f"Files in {TEMP_DIR}: {downloaded_files}")

        # Find the video file and delete unwanted files
        video_file = None
        for file in downloaded_files:
            file_path = os.path.join(TEMP_DIR, file)
            if file.endswith('.mp4'):  # Keep the video file
                video_file = file_path
            else:
                os.remove(file_path)  # Delete non-video files

        if video_file is None:
            raise ValueError(f"No video file found in {TEMP_DIR}.")

        print(f"Downloaded video file: {video_file}")
        return video_file

    except Exception as e:
        print(f"Error during video download: {e}")
        raise ValueError("Failed to download the reel.") from e

def extract_audio(video_path):
    try:
        # Check if the video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Specify the audio output path
        audio_path = video_path.replace(".mp4", ".wav")

        # Debug log to check paths
        print(f"Attempting to extract audio from: {video_path}")
        print(f"Audio will be saved to: {audio_path}")

        # Extract audio
        clip = VideoFileClip(video_path)  # Ensure you're using VideoFileClip, not AudioFileClip
        clip.audio.write_audiofile(audio_path)
        clip.close()

        print(f"Audio extraction successful: {audio_path}")
        return audio_path
    except Exception as e:
        print(f"Error during audio extraction: {e}")
        raise ValueError("Failed to extract audio from video.") from e

def transcribe_audio(audio_path):
    """Transcribe audio using SpeechRecognition."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return "Could not understand audio."
    except Exception as e:
        raise ValueError("Failed to transcribe audio.") from e

def cleanup_temp_dir():
    """Cleanup temporary files in TEMP_DIR."""
    print("Cleaning up temporary files...")
    for file in os.listdir(TEMP_DIR):
        file_path = os.path.join(TEMP_DIR, file)
        os.remove(file_path)
    print("Temporary files cleaned up.")

# Register the cleanup function to run on server exit
atexit.register(cleanup_temp_dir)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract-info', methods=['POST'])
def extract_info():
    try:
        data = request.json
        link = data.get('link', '')

        # Validate the link
        if "instagram.com" not in link:
            return jsonify({"info": "Invalid Instagram link."}), 400

        # Step 1: Download the reel
        video_path = download_instagram_reel(link)

        # Step 2: Extract audio
        audio_path = extract_audio(video_path)

        # Step 3: Transcribe audio
        transcription = transcribe_audio(audio_path)

        # Clean up the specific files used in this request
        # os.remove(video_path)
        # os.remove(audio_path)

        return jsonify({"info": transcription})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Shared progress dictionary
progress = {"status": "idle", "percent": 0}

def update_progress(status, percent):
    """Update the shared progress dictionary."""
    progress["status"] = status
    progress["percent"] = percent

def long_running_task():
    """Example task to simulate progress updates."""
    update_progress("Downloading video", 10)
    time.sleep(1)
    update_progress("Extracting audio", 50)
    time.sleep(1)
    update_progress("Transcribing audio", 90)
    time.sleep(1)
    update_progress("Completed", 100)

@app.route('/start-task', methods=['POST'])
def start_task():
    """Start a long-running task."""
    thread = threading.Thread(target=long_running_task)
    thread.start()
    return jsonify({"message": "Task started."})

@app.route('/progress', methods=['GET'])
def get_progress():
    """Get the progress of the task."""
    return jsonify(progress)

if __name__ == '__main__':
    app.run(debug=True)
