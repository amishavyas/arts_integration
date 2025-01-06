from flask import Flask, request, jsonify, Blueprint
from audio_processor import AudioProcessor
from session_manager import setup_session

app = Flask(__name__)
audio_bp = Blueprint('audio', __name__)

# Set up session directories and initialize AudioProcessor with paths
session_dir, audio_dir = setup_session()
audio_processor = AudioProcessor(session_dir=session_dir, audio_dir=audio_dir)

@audio_bp.route('/start_recording', methods=['POST'])
def start_recording():
    try:
        audio_processor.start_recording()
        return jsonify({"status": "success", "message": "Recording started"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@audio_bp.route('/stop_recording', methods=['POST'])
def stop_recording():
    try:
        audio_processor.stop_recording()
        return jsonify({"status": "success", "message": "Recording stopped"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@audio_bp.route('/update_image', methods=['POST'])
def update_image():
    try:
        data = request.get_json()
        image_id = data.get('image_id')
        audio_processor.update_current_image(image_id)
        return jsonify({"status": "success", "message": "Image updated"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


app.register_blueprint(audio_bp)