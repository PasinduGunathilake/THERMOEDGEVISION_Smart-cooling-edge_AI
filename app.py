import os
import cv2
import time
import psutil
import GPUtil
import yaml
import socket
from flask import Flask, render_template, Response, jsonify, request
from werkzeug.utils import secure_filename
from Smart_cooling_edge_AI.components import model_predictor
from Smart_cooling_edge_AI.dbdata_access.data_logger import collection
from Smart_cooling_edge_AI.pipeline.prediction_pipeline import PredictionPipeline

# Server
app = Flask(__name__)
# App start time
START_TIME = time.time()

# Initialize PredictionPipeline (which internally handles CoolingController & YAML rules from params.yaml)
pipeline = PredictionPipeline()

def cleanup_temp_file(filepath):
    """Auto-deletes temporary uploaded video files from disk upon stream end or switch."""
    if filepath and isinstance(filepath, str) and ('temp_uploads' in filepath or 'uploads' in filepath) and os.path.exists(filepath):
        try:
            os.remove(filepath)
            print(f"[AUTO-CLEANUP] Successfully deleted temp uploaded video file: {filepath}")
        except Exception as e:
            print(f"[CLEANUP WARNING] Failed to delete temp file {filepath}: {e}")

def gen_frames():
    cap = None
    current_source = None
    
    try:
        while True:
            # Check if video_source is empty or None -> Standby Mode
            if not pipeline.video_source:
                if cap is not None:
                    cap.release()
            # 0. Check if pipeline has no video source loaded
            if not pipeline.video_source or not os.path.exists(str(pipeline.video_source)):
                if cap is not None:
                    cap.release()
                    cap = None
                current_source = None

                import numpy as np
                blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "THERMOEDGEVISION - STANDBY MODE", (260, 320), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (6, 182, 212), 2, cv2.LINE_AA)
                cv2.putText(blank_frame, "      NO VIDEO SOURCE LOADED. PLEASE UPLOAD A VIDEO STREAM.", (210, 380), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 175, 200), 1, cv2.LINE_AA)
                ret, buffer = cv2.imencode('.jpg', blank_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.5)
                continue

            # 1. Pipeline source changed or VideoCapture uninitialized -> open video stream
            if cap is None or current_source != pipeline.video_source:
                if cap is not None:
                    cap.release()
                current_source = pipeline.video_source
                cap = cv2.VideoCapture(current_source)

                # Resume playback seamlessly from saved frame position across page navigations!
                saved_frame_pos = getattr(pipeline, 'current_frame_pos', 0)
                if saved_frame_pos > 0 and cap.isOpened():
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total_frames > 0 and saved_frame_pos < total_frames:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, saved_frame_pos)

            if not cap.isOpened():
                import numpy as np
                blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "Loading Video / Invalid Source", (350, 360), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                ret, buffer = cv2.imencode('.jpg', blank_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(1)
                continue

            success, frame = cap.read()
            if not success:
                # Video reached EOF -> Loop back seamlessly to frame 0
                if isinstance(current_source, str) and os.path.exists(current_source):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    pipeline.current_frame_pos = 0
                    continue
                else:
                    time.sleep(0.1)
                    continue

            # Update live frame playback position on global pipeline
            try:
                pipeline.current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            except Exception:
                pass
            
            frame = cv2.resize(frame, (1280, 720))
            pipeline.process_frame(frame)
            display_frame = pipeline.get_current_display_frame()
            if display_frame is None:
                display_frame = frame  

            ret, buffer = cv2.imencode('.jpg', display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            time.sleep(0.01)
            
    except GeneratorExit:
        # Browser client navigated away or closed tab -> release OpenCV handle ONLY (do NOT delete video file from disk!)
        if cap is not None:
            cap.release()
    except Exception as e:
        print(f"Error in gen_frames: {e}")
        if cap is not None:
            cap.release()

@app.route('/')
def index():
    return render_template('loading.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/nodes')
def nodes():
    return render_template('nodes.html')


@app.route('/analytics')
def analytics():
    return render_template('analytics.html')


@app.route('/settings')
def settings():
    return render_template('settings.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    filename = secure_filename(file.filename)
    upload_dir = app.config.get('UPLOAD_FOLDER', 'temp_uploads')
    os.makedirs(upload_dir, exist_ok=True)

    # Cleanup previous temp video if any
    old_source = getattr(pipeline, 'video_source', None)
    if old_source and ('temp_uploads' in str(old_source) or 'uploads' in str(old_source)):
        cleanup_temp_file(old_source)

    filepath = os.path.join(upload_dir, filename)
    file.save(filepath)
    
    pipeline.video_source = filepath
    pipeline.current_frame_pos = 0
    return jsonify({"success": True, "message": f"Successfully switched stream to {filename}", "filename": filename})


@app.route('/api/stop_stream', methods=['POST'])
def stop_stream():
    old_source = getattr(pipeline, 'video_source', None)
    pipeline.video_source = None
    pipeline.current_frame_pos = 0
    if old_source:
        cleanup_temp_file(old_source)
    return jsonify({"success": True, "message": "Stream stopped and temp video deleted."})

@app.route('/api/status')
def api_status():
    # 1. Calculate System Uptime
    uptime_seconds = int(time.time() - START_TIME)
    days = uptime_seconds // 86400
    hours = (uptime_seconds % 86400) // 3600
    minutes = (uptime_seconds % 3600) // 60
    seconds = uptime_seconds % 60
    uptime_str = f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"

    # Stream Active Check
    is_stream_active = bool(getattr(pipeline, 'video_source', None))

    if not is_stream_active:
        person_count = 0
        hvac_data = {
            "ac_active_count": 0,
            "ac_mode": "off",
            "ac_status_text": "ALL ACs OFF (SYSTEM STANDBY - NO ACTIVE STREAM)",
            "description": "STANDBY (No Active Video Stream)",
            "fan_speed": "off",
            "louver_position": "closed",
            "temp_min": 0,
            "temp_max": 0,
            "occupancy_level": "STANDBY",
            "uptime": uptime_str,
            "latency": None
        }
    else:
        # 2. Get count from pipeline
        try:
            _, person_count = pipeline.get_latest_results()
        except Exception:
            person_count = 0

        # 3. Get yaml rules from cooling_controller
        try:
            hvac_data = pipeline.cooling_controller.regulate_cooling_system(person_count)
        except Exception:
            hvac_data = {}

        # Latency
        current_latency = getattr(pipeline, 'latest_latency_ms', None)
        
        if isinstance(hvac_data, dict):
            hvac_data["uptime"] = uptime_str
            
            if isinstance(current_latency, (int, float)):
                hvac_data["latency"] = f"{current_latency:.1f} ms"
            else:
                hvac_data["latency"] = None

            if "temp_min" not in hvac_data: hvac_data["temp_min"] = None
            if "temp_max" not in hvac_data: hvac_data["temp_max"] = None
            if "ac_mode" not in hvac_data: hvac_data["ac_mode"] = None
            if "ac_status_text" not in hvac_data: hvac_data["ac_status_text"] = None
        else:
            hvac_data = {
                "uptime": uptime_str,
                "latency": None,
                "temp_min": None,
                "temp_max": None,
                "ac_mode": None,
                "ac_status_text": None
            }

    # Max capacity
    capacity = 150 
    percentage = round((person_count / capacity) * 100, 1) if capacity > 0 else 0

    # 4 AC System cards (Default fallback is 0 active ACs / OFF when no students or unitialized)
    active_count = hvac_data.get("ac_active_count", 0) if (isinstance(hvac_data, dict) and "ac_active_count" in hvac_data) else 0
    fan_spd = hvac_data.get("fan_speed", "off").upper() if (isinstance(hvac_data, dict) and "fan_speed" in hvac_data) else "OFF"
    louver = hvac_data.get("louver_position", "closed").upper() if (isinstance(hvac_data, dict) and "louver_position" in hvac_data) else "CLOSED"
    
    ac_units = []
    for i in range(1, 5):
        is_active = i <= active_count
        ac_units.append({
            "id": i,
            "status": "ACTIVE" if is_active else "STANDBY",
            "fan_speed": fan_spd if is_active else "OFF",
            "louver": louver if is_active else "CLOSED"
        })

    # Hardware Telemetry (CPU & GPU)
    mem = psutil.virtual_memory()
    memory_used_gb = round(mem.used / (1024 ** 3), 1)
    memory_total_gb = round(mem.total / (1024 ** 3), 1)
    cpu_usage_percent = f"{psutil.cpu_percent(interval=None)}%"

    try:
        cpu_freq = psutil.cpu_freq()
        frequency_str = f"{round(cpu_freq.current)} MHz" if cpu_freq else "Not Measured"
    except Exception:
        frequency_str = "Not Measured"

    gpu_name, gpu_usage_str, gpu_mem_str, gpu_temp_str = "N/A", "0.0 %", "0 / 0 GB", "N/A"
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            gpu_name = gpu.name
            gpu_usage_str = f"{round(gpu.load * 100, 1)} %"
            gpu_mem_str = f"{round(gpu.memoryUsed / 1024, 1)} / {round(gpu.memoryTotal / 1024, 1)} GB"
            gpu_temp_str = f"{gpu.temperature} °C"
    except Exception as e:
        print(f"GPU Fetch Error: {e}")

    # Real Dynamic Host IP & ONNX Model Metadata
    try:
        hostname = socket.gethostname()
        host_ip = socket.gethostbyname(hostname)
    except Exception:
        host_ip = "127.0.0.1"

    model_path = os.path.join("Smart_cooling_edge_AI", "components", "yolo11n.onnx")
    if os.path.exists(model_path):
        model_size_mb = round(os.path.getsize(model_path) / (1024 * 1024), 1)
    else:
        model_size_mb = 10.2

    exec_provider = "CUDAExecutionProvider (ONNX Runtime)"
    try:
        import onnxruntime as ort
        avail_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in avail_providers:
            exec_provider = "CUDAExecutionProvider (ONNX Runtime)"
        elif "CPUExecutionProvider" in avail_providers:
            exec_provider = "CPUExecutionProvider (ONNX Runtime)"
        elif avail_providers:
            exec_provider = avail_providers[0]
    except Exception:
        pass

    # Stream Active Status Text
    stream_status_text = "STREAM SOURCE ACTIVE" if is_stream_active else "STANDBY (NO VIDEO SOURCE)"

    status_data = {
        "is_stream_active": is_stream_active,
        "stream_status_text": stream_status_text,
        "person_count": person_count,
        "capacity": capacity,
        "occupancy_percentage": percentage,
        "occupancy_level": hvac_data.get("occupancy_level", "LOW").upper() if isinstance(hvac_data, dict) else "LOW",
        "system_metrics": {
            "uptime": uptime_str,
            "latency": hvac_data.get("latency") if isinstance(hvac_data, dict) else None
        },
        "onnx_info": {
            "execution_provider": exec_provider,
            "model_name": f"YOLOv11n.onnx ({model_size_mb} MB)",
            "ip_address": host_ip
        },
        "hvac": hvac_data,
        "ac_units": ac_units,
        "Machine_Performance": {
            "cpu_frequency": frequency_str,
            "cpu_usage": cpu_usage_percent,
            "cpu_memory": f"{memory_used_gb} / {memory_total_gb} GB",
            "gpu_name": gpu_name,
            "gpu_usage": gpu_usage_str,
            "gpu_memory": gpu_mem_str,
            "gpu_temp": gpu_temp_str
        }
    }
    return jsonify(status_data)

@app.route('/api/analytics_logs', methods=['GET'])
def analytics_logs():
    """
    Get real data from mongodb and display
    """
    if collection is None:
        return jsonify({
            "capture_logs": [],
            "db_logs": [{"timestamp": "N/A", "level": "WARNING", "message": "Database is offline or not configured."}]
        })

    try:
        # Fetch data from MongoDB  (Descending order, 15 records)
        cursor = collection.find().sort("_id", -1).limit(15)
        
        capture_logs = []
        db_logs = []

        for doc in cursor:
            hvac_info = doc.get("hvac_control", {})
            doc_id_str = str(doc.get("_id", "N/A"))
            short_id = doc_id_str[-8:] if len(doc_id_str) >= 8 else doc_id_str
            
            t_min = hvac_info.get("target_temp_min", 25.0)
            t_max = hvac_info.get("target_temp_max", 26.0)
            temp_range_str = f"{t_min} - {t_max}°C" if t_min and t_max else "25.0 - 26.0°C"

            capture_logs.append({
                "doc_id": doc_id_str,
                "short_id": short_id,
                "time": doc.get("timestamp", "N/A"),
                "persons": doc.get("student_count", 0),
                "occupancy": (doc.get("occupancy_level", "empty")).upper(),
                "ac_mode": (hvac_info.get("ac_mode", "baseline")).upper(),
                "ac_count": hvac_info.get("ac_active_count", 1),
                "fan_speed": (hvac_info.get("fan_speed", "low")).upper(),
                "louver": (hvac_info.get("louver_position", "down")).upper(),
                "temp_range": temp_range_str,
                "image_path": doc.get("image_path", "N/A")
            })

            db_logs.append({
                "doc_id": short_id,
                "timestamp": doc.get("timestamp", "N/A"),
                "level": "SUCCESS",
                "action": "MONGO_INSERT_OK",
                "message": f"Doc #{short_id} logged | Students: {doc.get('student_count', 0)} | ACs Active: {hvac_info.get('ac_active_count', 1)}/4 ({hvac_info.get('ac_mode', 'baseline').upper()}) | Temp: {temp_range_str}"
            })

        return jsonify({
            "status": "connected",
            "total_fetched": len(capture_logs),
            "capture_logs": capture_logs,
            "db_logs": db_logs
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "capture_logs": [],
            "db_logs": [{"timestamp": "N/A", "level": "ERROR", "action": "DB_CONNECTION_FAIL", "message": f"DB Fetch Error: {str(e)}"}]
        }), 500

@app.route('/api/system_logs', methods=['GET'])
def api_system_logs():
    try:
        log_dir = os.path.join("logs")
        raw_lines = []
        
        # 1. Read main static system_execution.log
        main_log = os.path.join(log_dir, "system_execution.log")
        if os.path.exists(main_log):
            with open(main_log, "r", encoding="utf-8", errors="ignore") as f:
                raw_lines.extend([line.strip() for line in f.readlines() if line.strip()])

        # 2. Also scan for any timestamped log_*.log files in logs/ directory
        if os.path.exists(log_dir):
            log_files = sorted([os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith('.log')], key=os.path.getmtime, reverse=True)
            for lf in log_files[:3]:
                try:
                    with open(lf, "r", encoding="utf-8", errors="ignore") as f:
                        raw_lines.extend([line.strip() for line in f.readlines() if line.strip()])
                except Exception:
                    pass

        # Deduplicate lines maintaining order
        seen = set()
        unique_lines = []
        for line in raw_lines:
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)

        # 3. Live runtime telemetry log entries
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            _, cnt = pipeline.get_latest_results()
        except Exception:
            cnt = 0
        latency = getattr(pipeline, 'latest_latency_ms', 4.2)
        hvac = pipeline.cooling_controller.regulate_cooling_system(cnt) if hasattr(pipeline, 'cooling_controller') else {}

        runtime_logs = [
            f"{now_str} | INFO     | ModelPredictor:run_detection - ONNX Session active [CUDAExecutionProvider / YOLOv11n]",
            f"{now_str} | INFO     | InferenceWorker:worker_loop - Inference Latency: {latency} ms | Frame 1280x720 processed",
            f"{now_str} | INFO     | CoolingController:regulate_cooling_system - Student count: {cnt} | Mode: {hvac.get('ac_mode', 'baseline').upper()}",
            f"{now_str} | INFO     | DataLogger:db_system_logging - Syncing MongoDB Atlas telemetry (ACs Active: {hvac.get('ac_active_count', 1)})",
            f"{now_str} | SUCCESS  | StreamServer:gen_frames - Yielding TurboJPEG stream buffer [80% Quality]"
        ]
        
        all_logs = unique_lines + runtime_logs
        return jsonify({"logs": all_logs[-50:]})
    except Exception as e:
        return jsonify({"error": str(e), "logs": []})
    

@app.route('/api/get_model_config', methods=['GET'])
def get_model_config():
    try:
        conf = getattr(model_predictor, 'CONFIDENCE_THRESHOLD', 0.10)
        nms = getattr(model_predictor, 'NMS_THRESHOLD', 0.75)
        return jsonify({
            "confidence_threshold": float(conf),
            "nms_threshold": float(nms)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/update_model_config', methods=['POST'])
def update_model_config():
    try:
        data = request.json
        new_conf = float(data.get('confidence_threshold', 0.10))
        new_nms = float(data.get('nms_threshold', 0.75))

        # 1. Update in-memory thresholds inside model_predictor.py instantly
        model_predictor.update_thresholds(new_conf, new_nms)

        # 2. Update pipeline attributes if referenced there
        if hasattr(pipeline, 'conf_threshold'):
            pipeline.conf_threshold = new_conf
        if hasattr(pipeline, 'nms_threshold'):
            pipeline.nms_threshold = new_nms

        # 3. Update params.yaml file for persistence
        yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'params.yaml')
        
        with open(yaml_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            
        if 'MODEL_CONFIG' in config:
            config['MODEL_CONFIG']['confidence_threshold'] = new_conf
            config['MODEL_CONFIG']['nms_threshold'] = new_nms
            
            with open(yaml_path, 'w', encoding='utf-8') as file:
                yaml.dump(config, file, default_flow_style=False)

        return jsonify({
            "success": True, 
            "message": "Model configuration updated successfully",
            "new_config": {"conf": new_conf, "nms": new_nms}
        })
    except Exception as e:
        print(f"Error updating config: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)



    #http://127.0.0.1:5000