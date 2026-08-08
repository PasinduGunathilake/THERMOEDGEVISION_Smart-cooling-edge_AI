import cv2
import time
from pathlib import Path
from Smart_cooling_edge_AI.components.model_predictor import run_detection
from Smart_cooling_edge_AI.components.cooling_controller import CoolingController
from Smart_cooling_edge_AI.utils.image_utils import save_annotated_frame
from Smart_cooling_edge_AI.dbdata_access.data_logger import db_system_logging
from Smart_cooling_edge_AI.utils.configurations import read_yaml
from Smart_cooling_edge_AI.logging.logger import get_logger


# Logger initialization
logger = get_logger("PredictionPipeline")



class PredictionPipeline:
    def __init__(self, video_source=None, interval_seconds=None):
        try:
            project_root = Path(__file__).resolve().parent.parent.parent
            config = read_yaml(project_root / "params.yaml")
            logger.info("Configuration parameters loaded successfully in PredictionPipeline.")
            
            # Extract configuration parameters safely with fallback defaults to ensure production resilience and prevent KeyError crashes
            pipeline_config = config.get('PIPELINE_CONFIG', {})
            
            self.video_source = video_source if video_source is not None else pipeline_config.get('video_source', 0)
            self.interval_seconds = interval_seconds if interval_seconds is not None else pipeline_config.get('interval_seconds', 300)
            self.cooling_controller = CoolingController()
            
            logger.info(f"PredictionPipeline initialized with video source: {self.video_source} and interval: {self.interval_seconds}s")
            
        except Exception as e:
            logger.critical(f"Failed to initialize PredictionPipeline: {str(e)}", exc_info=True)
            self.video_source = 0
            self.interval_seconds = 5
            self.cooling_controller = CoolingController()



    def run_pipeline(self):
        cap = None
        try:
            logger.info(f"Attempting to open video source: {self.video_source}")
            cap = cv2.VideoCapture(self.video_source)
            
            if not cap.isOpened():
                logger.critical(f"Error: Could not open video source at {self.video_source}.")
                return

            logger.info("Starting smart classroom cooling system pipeline. Press 'q' in the video window to exit.")
            last_checked_time = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Video stream ended or failed to grab frame from source.")
                    break
                    
                current_time = time.time()
                
                # Run inference and logging based on interval
                if current_time - last_checked_time >= self.interval_seconds or last_checked_time == 0:
                    logger.info("Interval reached. Running AI detection & cooling control adjustment...")
                    
                    try:
                        detected_boxes, person_count = run_detection(frame)
                        hvac_data = self.cooling_controller.regulate_cooling_system(person_count)
                        image_path, timestamp_str = save_annotated_frame(frame, detected_boxes, person_count, hvac_data)
                        db_system_logging(person_count, hvac_data, image_path, timestamp_str)
                    except Exception as e:
                        logger.error(f"Error occurred during interval processing block: {str(e)}", exc_info=True)
                        
                    last_checked_time = current_time

                # Real-time view updates
                try:
                    display_frame = frame.copy()
                    detected_boxes, person_count = run_detection(frame)
                    
                    for box in detected_boxes:
                        x1, y1, x2, y2 = map(int, box[:4])
                        conf = box[4] if len(box) > 4 else 0.0
                        
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        label = f"confidence: {conf:.2f}"
                        font_scale = 0.6
                        font_thickness = 2
                        
                        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                        
                        box_x1 = x1
                        box_y1 = y1 - text_height - 10
                        box_x2 = x1 + text_width + 5
                        box_y2 = y1
                        
                        cv2.rectangle(display_frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 255), -1)
                        
                        cv2.putText(
                            display_frame, 
                            label, 
                            (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            font_scale, 
                            (255, 255, 255), 
                            font_thickness, 
                            cv2.LINE_AA
                        )

                    cv2.imshow("Video Analysis", display_frame)

                except Exception as e:
                    logger.error(f"Error occurred during real-time frame rendering/display: {str(e)}", exc_info=True)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User requested termination ('q' pressed). Exiting pipeline loop.")
                    break

        except Exception as e:
            logger.critical(f"Critical error encountered in pipeline execution: {str(e)}", exc_info=True)
        finally:
            if cap is not None and cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()
            logger.info("Video stream closed and system shut down safely.")