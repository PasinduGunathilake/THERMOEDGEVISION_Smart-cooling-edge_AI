import time
import threading
from concurrent.futures import ThreadPoolExecutor
from Smart_cooling_edge_AI.components.model_predictor import run_detection
from Smart_cooling_edge_AI.components.cooling_controller import CoolingController
from Smart_cooling_edge_AI.utils.image_utils import save_annotated_frame
from Smart_cooling_edge_AI.dbdata_access.data_logger import db_system_logging
from Smart_cooling_edge_AI.logging.logger import get_logger


logger = get_logger("InferenceWorker")


class InferenceWorker:
    """
    Background YOLO inference thread - fully self-contained.

    Runs run_detection() continuously in a daemon thread so the main
    display loop is never blocked by heavy ONNX computation.

    Public API (called by PredictionPipeline):
        worker.set_frame(frame)   -> push the latest camera frame
        worker.get_results()      -> returns (boxes, count) from last inference
        worker.start()            -> launches the daemon thread
        worker.stop()             -> signals the thread to exit cleanly

    All threading primitives (Lock, Thread) are internal - callers never
    touch raw thread mechanics.
    """

    def __init__(self, cooling_controller: CoolingController, interval_seconds: int):
        self._cooling_controller  = cooling_controller
        self._interval_seconds    = interval_seconds
        self._executor = ThreadPoolExecutor(max_workers=2)
        

        # Internal shared state
        self._latest_frame  = None
        self._latest_boxes  = []
        self._latest_count  = 0
        self._latest_latency_ms = 0.0

        # Thread synchronisation
        self._frame_lock    = threading.Lock()
        self._result_lock   = threading.Lock()
        self._running       = False
        self._thread        = None


    # Public API
    def set_frame(self, frame) -> None:
        """Called by the main thread -- updates the frame to be processed next."""
        with self._frame_lock:
            self._latest_frame = frame

    def get_results(self) -> tuple:
        """Called by the main thread -- returns (boxes, count) from latest inference."""
        with self._result_lock:
            return list(self._latest_boxes), self._latest_count

    def get_latency(self) -> float:
        """Called by main thread -- returns latest inference latency in ms."""
        with self._result_lock:
            return self._latest_latency_ms

    def start(self) -> None:
        """Launches the background inference daemon thread."""
        self._running = True
        self._thread  = threading.Thread(
            target=self._worker_loop,
            name="InferenceWorker",
            daemon=True
        )
        self._thread.start()
        logger.info("InferenceWorker thread started.")

    def stop(self) -> None:
        """Signals the worker thread to stop after its current inference cycle."""
        self._running = False
        logger.info("InferenceWorker thread stop signal sent.")




    # Internal worker loop
    def _worker_loop(self) -> None:
        """
        Core inference loop. Runs continuously in the background:
          1. Grabs the latest frame pushed by the main thread.
          2. Runs YOLO ONNX inference.
          3. Updates shared detection results for the display thread.
          4. Every interval_seconds: HVAC control, frame save, MongoDB log.
        """
        last_logged_time = 0

        while self._running:
            # Grab latest frame safely
            with self._frame_lock:
                frame = self._latest_frame

            if frame is None:
                time.sleep(0.01)
                continue

            # YOLO inference timing
            t0 = time.time()
            try:
                detected_boxes, person_count = run_detection(frame)
            except Exception as e:
                logger.error(f"run_detection failed: {str(e)}", exc_info=True)
                detected_boxes, person_count = [], 0
            t1 = time.time()
            latency_ms = round((t1 - t0) * 1000, 1)

            # Update shared results for display thread
            with self._result_lock:
                self._latest_boxes = detected_boxes
                self._latest_count = person_count
                self._latest_latency_ms = latency_ms

            # Interval: cooling control + frame save + DB log
            current_time = time.time()
            if current_time - last_logged_time >= self._interval_seconds or last_logged_time == 0:
                logger.info("Interval reached. Running cooling control & data logging...")

                try:
                    hvac_data = self._cooling_controller.regulate_cooling_system(person_count)
                    image_path, timestamp_str = save_annotated_frame(frame, detected_boxes, person_count, hvac_data)
                    # Main AI thread runs without blocking by submitting DB logging to background executor thread pool
                    self._executor.submit(db_system_logging, person_count, hvac_data, image_path, timestamp_str)
                except Exception as e:
                    logger.error(f"Error during interval processing: {str(e)}", exc_info=True)

                last_logged_time = current_time

        logger.info("InferenceWorker thread exited cleanly.")
