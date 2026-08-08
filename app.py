import cv2
import yaml
import numpy as np
from pathlib import Path
from flask import Flask, render_template, Response, jsonify
from Smart_cooling_edge_AI.pipeline.prediction_pipeline import PredictionPipeline
from Smart_cooling_edge_AI.logging.logger import get_logger

# Initialize logger for the Flask server tracking
logger = get_logger("FlaskServer")

# Initialize Flask application instance
app = Flask(__name__)

# Locate and load configuration parameters from params.yaml
current_dir = Path(__file__).resolve().parent
config_path = current_dir / "params.yaml"

try:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
except Exception as e:
    logger.error(f"Failed to load params.yaml in app.py: {e}")
    config = {}

# Initialize the core Prediction Pipeline (manages background inference worker and HVAC control)
pipeline = PredictionPipeline()

@app.route("/")
def index():
    """Render the main HTML monitoring dashboard."""
    return render_template("index.html")

def generate_frames():
    """
    Generator function that continuously fetches the latest annotated frame 
    from the prediction pipeline and yields it as a JPEG byte stream for live streaming.
    """
    global pipeline
    while True:
        # Fetch the latest processed frame containing bounding boxes from the pipeline
        frame = pipeline.get_current_display_frame()
        if frame is None:
            # Fallback to a blank black frame if no frame is available yet
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Encode the frame into JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # Yield the frame using multipart streaming format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route("/video_feed")
def video_feed():
    """Video streaming route linked to the web dashboard UI tag."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/api/status")
def api_status():
    """API endpoint to fetch real-time telemetry (current person count, system status, active model)."""
    boxes, count = pipeline.get_latest_results()
    return jsonify({
        "person_count": count,
        "status": "Running",
        "model": config.get("MODEL_CONFIG", {}).get("model_name", "YOLOv11")
    })

if __name__ == "__main__":
    # Extract server configurations with safe fallbacks
    flask_config = config.get("FLASK_CONFIG", {})
    host = flask_config.get("host", "0.0.0.0")
    port = flask_config.get("port", 5001)
    
    logger.info(f"Starting Flask server on http://{host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)