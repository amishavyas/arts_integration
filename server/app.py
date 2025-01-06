from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS
from audio_endpoints import audio_bp

app = Flask(__name__)
CORS(app)
app.register_blueprint(audio_bp)

@app.route('/reset_img_data', methods=['GET', 'POST'])
def reset_img_data():
    return jsonify({"status": "success"}), 200

# Add error handling for missing routes
@app.route('/<path:path>')
def catch_all(path):
    return '', 204  # Return empty response with 204 status for unhandled routes

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

