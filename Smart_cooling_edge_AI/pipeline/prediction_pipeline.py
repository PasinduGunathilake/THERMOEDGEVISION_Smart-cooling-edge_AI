import cv2
from pathlib import Path
from Smart_cooling_edge_AI.components.cooling_controller import CoolingController
from Smart_cooling_edge_AI.pipeline.inference_thread import InferenceWorker
from Smart_cooling_edge_AI.utils.image_utils import draw_boxes
from Smart_cooling_edge_AI.utils.configurations import read_yaml
from Smart_cooling_edge_AI.logging.logger import get_logger

# Logger initialization
logger = get_logger("PredictionPipeline")

class PredictionPipeline:

    """
    Main pipeline -  orchestrates video capture, display, and AI inferenc.

      - Open the camera / video source.
      - Push frames to InferenceWorker (background thread - inference_thread.py).
      - Fetch latest bounding boxes and render them at full camera FPS.
      - Manage lifecycle (start / stop InferenceWorker cleanly).
    """

    def __init__(self, video_source=None, interval_seconds=None):
        try:
            project_root = Path(__file__).resolve().parent.parent.parent
            config = read_yaml(project_root / "params.yaml")
            logger.info("Configuration parameters loaded successfully in PredictionPipeline.")

            # Extract configuration parameters safely with fallback defaults
            pipeline_config = config.get('PIPELINE_CONFIG', {})

            self.video_source     = video_source     if video_source     is not None else pipeline_config.get('video_source', 0)
            self.interval_seconds = interval_seconds if interval_seconds is not None else pipeline_config.get('interval_seconds', 300)

            logger.info(f"PredictionPipeline initialized | source: {self.video_source} | interval: {self.interval_seconds}s")

        except Exception as e:
            logger.critical(f"Failed to initialize PredictionPipeline: {str(e)}", exc_info=True)
            self.video_source     = 0
            self.interval_seconds = 5

        # Initialise CoolingController and pass it to the worker
        self._cooling_controller = CoolingController()
        self._worker = InferenceWorker(
            cooling_controller=self._cooling_controller,
            interval_seconds=self.interval_seconds
        )


    def run_pipeline(self):
        
        """
        Main display loop.
        """
        cap = None
        try:
            logger.info(f"Attempting to open video source: {self.video_source}")
            cap = cv2.VideoCapture(self.video_source)

            if not cap.isOpened():
                logger.critical(f"Error: Could not open video source at {self.video_source}.")
                return

            # Start background inference thread , frame flow
            self._worker.start()
            logger.info("Smart classroom cooling system pipeline started. Press 'q' to exit.")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Video stream ended or failed to grab frame.")
                    break

                # Push latest frame to inference worker
                self._worker.set_frame(frame.copy()) 

                # Fetch latest detection results (uses last known boxes â never blocks)
                display_boxes, _ = self._worker.get_results()

                # Render bounding boxes on current frame and display
                try:
                    display_frame = draw_boxes(frame, display_boxes)
                    cv2.imshow("Video Analysis", display_frame)
                except Exception as e:
                    logger.error(f"Error during frame rendering: {str(e)}", exc_info=True)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User requested termination ('q' pressed).")
                    break

        except Exception as e:
            logger.critical(f"Critical error in pipeline execution: {str(e)}", exc_info=True)
        finally:
            self._worker.stop()
            if cap is not None and cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()
            logger.info("Video stream closed and system shut down safely.")
