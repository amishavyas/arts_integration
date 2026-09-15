import os
from flask import Blueprint, request, jsonify
from audio_processor import AudioProcessor, AudioConfig, find_scarlett_device
from session_manager import setup_session

# Create blueprint
audio_bp = Blueprint('audio', __name__)

TEST_MODE = os.environ.get("CONVO_RECORDER_TEST_MODE") == "1"

# Only touch the filesystem / audio hardware if a Scarlett is actually
# connected. Without it, the server still starts (so the frontend can show
# a "no audio device" warning) but no session folder is created and the
# recording endpoints are disabled.
device_index, device_info = find_scarlett_device()
DEVICE_CONNECTED = device_index is not None

audio_processor = None
if DEVICE_CONNECTED:
    session_dir, audio_dir = setup_session(test_mode=TEST_MODE)
    audio_processor = AudioProcessor(
        session_dir=session_dir,
        audio_dir=audio_dir,
        config=AudioConfig(device_index=device_index),
    )
else:
    print("WARNING: No Scarlett audio interface detected. "
          "Recording endpoints are disabled and no session folder was created.")

@audio_bp.route('/device_status', methods=['GET'])
def device_status():
    return jsonify({
        "connected": DEVICE_CONNECTED,
        "device_name": device_info["name"] if device_info else None,
        "test_mode": TEST_MODE,
    })

@audio_bp.route('/start_recording', methods=['POST'])
def start_recording():
    if audio_processor is None:
        return jsonify({"status": "error", "message": "No audio device connected"}), 503
    try:
        audio_processor.start_session()
        return jsonify({"status": "success", "message": "Recording started"})
    except Exception as e:
        print(f"Error starting recording: {str(e)}")  # Add debug logging
        return jsonify({"status": "error", "message": str(e)}), 500

@audio_bp.route('/stop_recording', methods=['POST'])
def stop_recording():
    if audio_processor is None:
        return jsonify({"status": "error", "message": "No audio device connected"}), 503
    try:
        audio_processor.stop_session()
        return jsonify({"status": "success", "message": "Recording stopped"})
    except Exception as e:
        print(f"Error stopping recording: {str(e)}")  # Add debug logging
        return jsonify({"status": "error", "message": str(e)}), 500

@audio_bp.route('/update_image', methods=['POST'])
def update_image():
    if audio_processor is None:
        return jsonify({"status": "error", "message": "No audio device connected"}), 503
    try:
        data = request.get_json()
        image_id = data.get('image_id')
        if not image_id:
            return jsonify({"status": "error", "message": "No image_id provided"}), 400
        audio_processor.update_current_image(image_id)
        return jsonify({"status": "success", "message": "Image updated"})
    except Exception as e:
        print(f"Error updating image: {str(e)}")  # Add debug logging
        return jsonify({"status": "error", "message": str(e)}), 500