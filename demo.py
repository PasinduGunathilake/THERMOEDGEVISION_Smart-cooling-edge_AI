import os
import cv2
from Smart_cooling_edge_AI.pipeline.prediction_pipeline import PredictionPipeline

def main():
    print("=" * 60)
    print("Running Smart Cooling Edge AI - Component & Pipeline Demo")
    print("=" * 60)

    model_path = "models/yolo11m.onnx"
    if not os.path.exists(model_path):
        print(f"Error: ONNX model not found at '{model_path}'.")
        print("Please place your 'yolo11m.onnx' file inside the 'models/' folder.")
        return

    # Initialize and run pipeline (using webcam 0, short interval for demo testing)
    video_source = 0 
    pipeline = PredictionPipeline(video_source=video_source, interval_seconds=300)
    
    try:
        pipeline.run_pipeline()
    except KeyboardInterrupt:
        print("\nDemo interrupted safely by user.")

if __name__ == "__main__":
    main()