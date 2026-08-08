import cv2
import threading
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
      - Flask Streaming
    """

    def __init__(self, video_source=None, interval_seconds=None):
        try:
            project_root = Path(__file__).resolve().parent.parent.parent
            config = read_yaml(project_root / "params.yaml")
            logger.info("Configuration parameters loaded successfully in PredictionPipeline.")

            # Extract configuration parameters safely with fallback defaults
            pipeline_config = config.get('PIPELINE_CONFIG', {})

            self.video_source = video_source if video_source  is not None else pipeline_config.get('video_source', 0)
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

        # Start background inference worker thread immediately
        self._worker.start()

        # Shared display frame state and threading lock for Flask streaming
        self._current_display_frame = None
        self._frame_lock = threading.Lock()





    def process_frame(self, frame) -> None:
        """
        Called by edge_streamer or external ingestion:
        1. Pushes the raw frame to the background inference worker.
        2. Fetches latest bounding boxes.
        3. Draws annotations and updates the cached display frame safely.
        """

        try:
            if frame is None:
                return

            # Push latest frame to inference worker (non-blocking)
            self._worker.set_frame(frame.copy())

            # Fetch latest detection results from brackground thread
            display_boxes, _ = self._worker.get_results()

            # Render bounding boxes on a copy of the frame for ui stream
            annotated_frame = draw_boxes(frame,display_boxes )

            # Safely update the shared display frame cache
            with self._frame_lock:# -------------------------------=============================================
                self._current_display_frame = annotated_frame

        except Exception as e:
            logger.error(f"Error processing frame in pipeline: {str(e)}", exc_info=True)







    def get_current_display_frame(self):
        """Called by Flask /video_feed route to stream the latest annotated frame."""
        with self._frame_lock:
            if self._current_display_frame is not None:
                return self._current_display_frame.copy()
            return None





    def get_latest_results(self) -> tuple:
        """Returns the latest (boxes, count) from the background worker."""
        return self._worker.get_results()


    def run_pipeline(self):
        
        """
        Main display loop (old).

        Standalone local execution loop (useful for direct testing without Flask).
        Opens the video source, pushes frames, and displays them via OpenCV window.

        """
        cap = None
        try:
            logger.info(f"Attempting to open video source: {self.video_source}")
            cap = cv2.VideoCapture(self.video_source)

            if not cap.isOpened():
                logger.critical(f"Error: Could not open video source at {self.video_source}.")
                return

            # Start background inference thread , frame flow
            ## self._worker.start() - old standalone system functons
            logger.info("Smart cooling system pipeline started locally. Press 'q' to exit.")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Video stream ended or failed to grab frame.")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue 
 
                # Push latest frame to inference worker
                ## self._worker.set_frame(frame.copy()) 

                # Fetch latest detection results (uses last known boxes â never blocks)
                ## display_boxes, _ = self._worker.get_results()

                # Render bounding boxes on current frame and display
                ##try:
                    ##display_frame = draw_boxes(frame, display_boxes)
                    ##cv2.imshow("Video Analysis", display_frame)
                ##except Exception as e:
                    ##logger.error(f"Error during frame rendering: {str(e)}", exc_info=True)

                # Process frame through pipeline
                self.process_frame(frame)

                # Fetch and display locally via OpenCV
                display_frame = self.get_current_display_frame()
                if display_frame is not None:
                    cv2.imshow("Video Analysis - Smart Cooling Edge AI", display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User requested termination ('q' pressed).")
                    break

        except Exception as e:
            logger.critical(f"Critical error in pipeline execution: {str(e)}", exc_info=True)
        finally:
            self.stop()
            if cap is not None and cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()
            logger.info("Video stream closed and system shut down safely.")






    def stop(self) -> None:
        """Cleanly stops the background worker thread."""
        logger.info("Stopping PredictionPipeline...")
        self._worker.stop()
        logger.info("PredictionPipeline stopped.")
