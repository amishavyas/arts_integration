from flask import Blueprint, request, jsonify
from audio_processor import AudioProcessor
from session_manager import setup_session

# Create blueprint
audio_bp = Blueprint('audio', __name__)

# Set up session directories and initialize AudioProcessor with paths
session_dir, audio_dir = setup_session()
audio_processor = AudioProcessor(session_dir=session_dir, audio_dir=audio_dir)

@audio_bp.route('/start_recording', methods=['POST'])
def start_recording():
    try:
        audio_processor.start_session()
        return jsonify({"status": "success", "message": "Recording started"})
    except Exception as e:
        print(f"Error starting recording: {str(e)}")  # Add debug logging
        return jsonify({"status": "error", "message": str(e)}), 500

@audio_bp.route('/stop_recording', methods=['POST'])
def stop_recording():
    try:
        audio_processor.stop_session()
        return jsonify({"status": "success", "message": "Recording stopped"})
    except Exception as e:
        print(f"Error stopping recording: {str(e)}")  # Add debug logging
        return jsonify({"status": "error", "message": str(e)}), 500

@audio_bp.route('/update_image', methods=['POST'])
def update_image():
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