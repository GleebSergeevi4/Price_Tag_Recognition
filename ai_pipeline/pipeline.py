"""
Video Processing Pipeline: Tracking → OCR → CSV Export
Loads video, tracks objects, crops them, recognizes text, saves to CSV
"""
import os
import cv2
import logging
from pathlib import Path
from collections import defaultdict
import pandas as pd
from rapidocr_onnxruntime import RapidOCR
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
STORAGE_DIR = PROJECT_ROOT / "storage"
VIDEOS_DIR = STORAGE_DIR / "videos"
CSV_DIR = STORAGE_DIR / "csv"
CROPS_DIR = STORAGE_DIR / "crops"
OCR_SRC_DIR = Path(__file__).parent / "ocr_src"
TRACKER_SRC_DIR = Path(__file__).parent / "tracker_src"

# Create output directory
CSV_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize models
logger.info("Initializing tracker...")
tracker = YOLO(str(TRACKER_SRC_DIR / "best.pt"))

logger.info("Initializing OCR...")
ocr = RapidOCR(
    det_model_path=str(OCR_SRC_DIR / "det.onnx"),
    rec_model_path=str(OCR_SRC_DIR / "rec.onnx"),
    rec_keys_path=str(OCR_SRC_DIR / "dict.txt"),
    use_angle_cls=True,
    angle_cls_model_path=str(OCR_SRC_DIR / "PP-LCNet_x1_0_doc_ori.onnx")
)


# STRIDE: take 1 frame out of every STRIDE frames per object
STRIDE = 6


def preprocess_for_ocr(image_np):
    """Preprocess crop image before OCR.

    `image_np` — numpy array in RGB format (H, W, C).
    By default this is a no-op. If you have a specific preprocessor,
    paste its body here and return the processed image (RGB numpy).
    """
    # Example (no-op): return input as-is
    return image_np



def extract_texts_from_ocr_result(ocr_result):
    """Extract recognized strings from RapidOCR result across common output shapes."""
    texts = []
    if not ocr_result:
        return texts

    for item in ocr_result:
        text = None

        # Common RapidOCR format: [box_points, text, conf] or [box_points, (text, conf)]
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            second = item[1]
            if isinstance(second, str):
                text = second
            elif isinstance(second, (list, tuple)) and len(second) > 0 and isinstance(second[0], str):
                text = second[0]

        # Fallback format: (text, conf)
        if text is None and isinstance(item, (list, tuple)) and len(item) >= 1 and isinstance(item[0], str):
            text = item[0]

        # Fallback format: plain string
        if text is None and isinstance(item, str):
            text = item

        if isinstance(text, str):
            cleaned = text.strip()
            if cleaned:
                texts.append(cleaned)

    return texts


def track_video(video_path, crops_video_dir):
    """
    Track objects in video and collect crops by object ID.
    
    Returns:
        dict: {object_id: [(frame_num, crop_path), ...]}
    """
    logger.info(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return {}
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video info: fps={fps:.1f}, total_frames={total_frames}")
    
    object_crops = defaultdict(list)  # {object_id: [(frame_num, crop_path), ...]}
    frame_num = 0
    tracked_objects = set()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run tracker
        results = tracker.track(frame, persist=True, conf=0.73, iou=0.23, verbose=False)
        
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i, box in enumerate(boxes):
                try:
                    # Get box coordinates and object ID
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    obj_id = int(box.id) if box.id is not None else -1
                    
                    if obj_id == -1:
                        continue
                    
                    tracked_objects.add(obj_id)
                    
                    # Crop object
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    
                    crop = frame[y1:y2, x1:x2]
                    
                    if crop.size > 0:
                        obj_dir = crops_video_dir / f"id_{obj_id}"
                        obj_dir.mkdir(parents=True, exist_ok=True)
                        crop_path = obj_dir / f"frame_{frame_num:06d}_{i:02d}.jpg"
                        if cv2.imwrite(str(crop_path), crop):
                            object_crops[obj_id].append((frame_num, crop_path))
                        else:
                            logger.warning(f"Failed to save crop for object {obj_id} at frame {frame_num}")
                
                except Exception as e:
                    logger.warning(f"Error processing box at frame {frame_num}: {e}")
                    continue
        
        frame_num += 1
        if frame_num % 30 == 0:
            logger.info(f"Processed {frame_num}/{total_frames} frames, tracked {len(tracked_objects)} objects")
    
    cap.release()
    logger.info(f"Tracking complete. Tracked {len(tracked_objects)} unique objects")
    return dict(object_crops)


def recognize_crops(object_crops):
    """
    Run OCR on object crops.
    
    Args:
        object_crops: {object_id: [(frame_num, crop_path), ...]}
    
    Returns:
        list: [{"object_id": int, "frame_num": int, "crop_path": str, "text": str}, ...]
    """
    logger.info(f"Starting OCR for {len(object_crops)} objects...")
    results = []
    total_crops = sum(len(crops) for crops in object_crops.values())
    processed = 0
    
    for obj_id, crops in sorted(object_crops.items()):
        for frame_num, crop_path in crops:
            try:
                # load image, preprocess, and send numpy RGB to OCR
                img = cv2.imread(str(crop_path))
                if img is None:
                    logger.warning("Failed to read crop image: %s", crop_path)
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pre = preprocess_for_ocr(img_rgb)

                ocr_result, elapsed = ocr(img_pre)

                # Extract text from OCR result
                text = ""
                if ocr_result:
                    texts = extract_texts_from_ocr_result(ocr_result)
                    text = " ".join(texts)

                results.append({
                    "object_id": obj_id,
                    "frame_num": frame_num,
                    "crop_path": str(crop_path),
                    "text": text
                })

                processed += 1
                if processed % 10 == 0:
                    logger.info(f"Processed {processed}/{total_crops} crops")

            except Exception as e:
                logger.warning(f"OCR error for object {obj_id}, frame {frame_num}: {e}")
                results.append({
                    "object_id": obj_id,
                    "frame_num": frame_num,
                    "crop_path": str(crop_path),
                    "text": ""
                })
    
    logger.info(f"OCR complete. Processed {processed} crops")
    return results


def save_results_csv(ocr_results, output_path):
    """Save OCR results to CSV."""
    df = pd.DataFrame(ocr_results)
    df = df.sort_values(['object_id', 'frame_num']).reset_index(drop=True)
    
    df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"Results saved to: {output_path}")
    
    # Print summary
    logger.info(f"\nSummary:")
    logger.info(f"  Total rows: {len(df)}")
    logger.info(f"  Unique objects: {df['object_id'].nunique()}")
    logger.info(f"  Non-empty text: {(df['text'].str.len() > 0).sum()}")


def process_video(video_path, output_csv_name=None):
    """
    Complete pipeline: track video, recognize text, save to CSV.
    
    Args:
        video_path: path to video file
        output_csv_name: output CSV filename (optional, auto-generated if None)
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        return

    crops_video_dir = CROPS_DIR / video_path.stem
    crops_video_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    if output_csv_name is None:
        output_csv_name = f"{video_path.stem}_ocr.csv"
    
    output_path = CSV_DIR / output_csv_name
    
    logger.info("=" * 60)
    logger.info(f"Pipeline start: {video_path.name}")
    logger.info("=" * 60)
    
    try:
        # Step 1: Track objects and collect crops
        logger.info("\n[Step 1/3] Tracking objects in video...")
        object_crops = track_video(video_path, crops_video_dir)
        
        if not object_crops:
            logger.warning("No objects tracked in video")
            return
        
        # Step 2: Sample crops (stride) and run OCR
        logger.info("\n[Step 2/3] Sampling crops with stride=%d and running OCR...", STRIDE)
        sampled = {}
        for obj_id, crops in object_crops.items():
            # keep first of every STRIDE crops for this object
            sampled[obj_id] = [c for idx, c in enumerate(crops) if idx % STRIDE == 0]
        total_before = sum(len(v) for v in object_crops.values())
        total_after = sum(len(v) for v in sampled.values())
        logger.info("Crops: before=%d, after_sampling=%d", total_before, total_after)
        ocr_results = recognize_crops(sampled)
        
        # Step 3: Save to CSV
        logger.info("\n[Step 3/3] Saving results to CSV...")
        save_results_csv(ocr_results, output_path)
        
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline complete!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)


def main():
    """Process all videos in storage/videos/ directory."""
    if not VIDEOS_DIR.exists():
        logger.error(f"Videos directory not found: {VIDEOS_DIR}")
        return
    
    video_files = list(VIDEOS_DIR.glob("*.mp4")) + list(VIDEOS_DIR.glob("*.avi"))
    
    if not video_files:
        logger.warning(f"No video files found in {VIDEOS_DIR}")
        return
    
    logger.info(f"Found {len(video_files)} video file(s) to process")
    
    for video_path in sorted(video_files):
        process_video(video_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Process specific video file
        video_path = sys.argv[1]
        output_csv = sys.argv[2] if len(sys.argv) > 2 else None
        process_video(video_path, output_csv)
    else:
        # Process all videos in storage/videos/
        main()
