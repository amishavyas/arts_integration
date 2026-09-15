from flask import Flask, request
from flask_cors import CORS
from audio_endpoints import audio_bp
import logging

# Set up logging - change to INFO level and format to be more concise
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Register the blueprint
app.register_blueprint(audio_bp)
logger.info("Audio blueprint registered")

# Remove the request logging middleware since it's too verbose
if __name__ == "__main__":
    logger.info("Starting Flask server on port 5001")
    app.run(host='0.0.0.0', port=5001, debug=False) 

