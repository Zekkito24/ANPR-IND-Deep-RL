import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort
from util import write_csv, get_car, read_license_plate
import add_missing_data
from src import visualize
from tqdm import tqdm

def _focus_score(bgr_crop: np.ndarray) -> float:
    """
    Blur/Sharpness proxy: variance of Laplacian.
    Returns a normalized score ~[0,1].
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return 0.0
    try:
        gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    except Exception:
        gray = bgr_crop if len(bgr_crop.shape) == 2 else None
    if gray is None or gray.size == 0:
        return 0.0
    v = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return float(max(0.0, min(1.0, 1.0 - np.exp(-v / 300.0))))

def run_pipeline(video_path: str, det_conf: float = 0.88, det_imgsz: int = 736, frame_skip: int = 3):
    """
    Main ANPR Execution Layer.
    Exposes det_conf and det_imgsz as actions for RL Meta-Optimization.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Professional Path Handling: Ensure models are found in /models folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    raw_csv = f"{video_name}_test.csv"
    interpolated_csv = f"{video_name}_interpolated.csv"
    output_video = f"{video_name}_output.mp4"
    detected_txt = f"{video_name}_detected_plates.txt"

    if os.path.exists(detected_txt):
        os.remove(detected_txt)

    results = {}
    mot_tracker = Sort()

    # Load models using relative paths for container portability
    coco_path = os.path.join(base_dir, 'models', 'yolov8n.pt')
    lpd_path = os.path.join(base_dir, 'models', 'INPD_more_accuracy_n.pt')
    
    coco_model = YOLO(coco_path)
    lpd_model = YOLO(lpd_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vehicles = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    frame_no = -1
    best_for_car = {}

    print(f"[INFO] Processing {video_name} with conf={det_conf} and imgsz={det_imgsz}")

    with tqdm(total=total_frames, desc="ANPR Pipeline", unit="frame") as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            pbar.update(1)

            # Hardware-Aware Optimization: Frame skipping
            if frame_no % frame_skip != 0:
                continue

            results[frame_no] = {}

            # Vehicle Detection
            det = coco_model(frame, conf=det_conf, imgsz=det_imgsz, verbose=False)[0]
            det_list = [
                [x1, y1, x2, y2, score]
                for x1, y1, x2, y2, score, class_id in det.boxes.data.tolist()
                if int(class_id) in vehicles
            ]
            det_arr = np.asarray(det_list) if det_list else np.empty((0, 5))

            # SORT Tracking
            track_ids = mot_tracker.update(det_arr)

            # License Plate Detection
            lp_res = lpd_model(frame, conf=0.22, imgsz=640, verbose=False)[0]

            for lp in lp_res.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = lp
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(lp, track_ids)
                if car_id == -1:
                    continue

                # Crop with padding for OCR robustness
                pad_y, pad_x = int((y2 - y1) * 0.15), int((x2 - x1) * 0.10)
                x1p, y1p = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
                x2p, y2p = min(frame.shape[1], int(x2 + pad_x)), min(frame.shape[0], int(y2 + pad_y))
                lp_cropped = frame[y1p:y2p, x1p:x2p]
                
                if lp_cropped.size == 0:
                    continue

                # OCR & Quality Scoring
                lp_text, lp_text_score = read_license_plate(lp_cropped)
                if lp_text is None:
                    continue

                sharp = _focus_score(lp_cropped)
                combined = float(lp_text_score) + 0.15 * float(score) + 0.10 * sharp

                # Maintain best read per track ID
                prev = best_for_car.get(int(car_id))
                if prev is None or combined > prev["combined"]:
                    best_for_car[int(car_id)] = {
                        "text": lp_text,
                        "text_score": float(lp_text_score),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "bbox_score": float(score),
                        "combined": float(combined)
                    }

                chosen = best_for_car[int(car_id)]
                results[frame_no][int(car_id)] = {
                    'car': {'bbox': [int(xcar1), int(ycar1), int(xcar2), int(ycar2)]},
                    'license_plates': {
                        'bbox': chosen["bbox"],
                        'text': chosen["text"],
                        'bbox_score': chosen["bbox_score"],
                        'text_score': chosen["text_score"]
                    }
                }

                with open(detected_txt, "a") as f:
                    f.write(f"Frame {frame_no}: {chosen['text']}\n")

    cap.release()

    # Step 2: Verification & Cleanup
    write_csv(results, raw_csv)
    add_missing_data.main(input_csv=raw_csv, output_csv=interpolated_csv)

    # Step 3: Visualization for Diagnostic Verification
    visualize.main(video_path=video_path, input_csv=interpolated_csv, output_path=output_video)
    print(f"[SUCCESS] Pipeline complete. Results saved to {output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge-Optimized ANPR Pipeline")
    parser.add_argument("--video_path", required=True, help="Path to input video")
    parser.add_argument("--det-conf", type=float, default=0.88, help="YOLO confidence (Action Space)")
    parser.add_argument("--det-imgsz", type=int, default=736, help="YOLO image size (Action Space)")
    parser.add_argument("--frame-skip", type=int, default=3, help="Hardware-aware frame skipping")
    args = parser.parse_args()
    
    run_pipeline(args.video_path, args.det_conf, args.det_imgsz, args.frame_skip)