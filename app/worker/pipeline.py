"""
Video Processing Pipeline: Tracking → OCR → CSV Export
Loads video, tracks objects, crops them, recognizes text, saves to CSV
Adapted for web service with async support.
"""
import logging
from pathlib import Path
from collections import defaultdict
from typing import Any

import cv2
import pandas as pd
from rapidocr_onnxruntime import RapidOCR
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# STRIDE: take 1 frame out of every STRIDE frames per object
STRIDE = 12

# Global model instances (lazy-loaded)
_tracker = None
_ocr = None
_models_dir = None


def set_models_dir(models_dir: Path) -> None:
    """Set the directory where ML models are stored."""
    global _models_dir
    _models_dir = Path(models_dir)


def get_tracker():
    """Lazy-load tracker model."""
    global _tracker
    if _tracker is None:
        if _models_dir is None:
            raise RuntimeError("Models dir not set. Call set_models_dir() first.")
        tracker_path = _models_dir / "tracker_src" / "best.pt"
        logger.info("Loading tracker from %s", tracker_path)
        _tracker = YOLO(str(tracker_path))
    return _tracker


def get_ocr():
    """Lazy-load OCR model."""
    global _ocr
    if _ocr is None:
        if _models_dir is None:
            raise RuntimeError("Models dir not set. Call set_models_dir() first.")
        ocr_dir = _models_dir / "ocr_src"
        logger.info("Loading OCR from %s", ocr_dir)
        _ocr = RapidOCR(
            det_model_path=str(ocr_dir / "det.onnx"),
            rec_model_path=str(ocr_dir / "rec.onnx"),
            rec_keys_path=str(ocr_dir / "dict.txt"),
            use_angle_cls=True,
            angle_cls_model_path=str(ocr_dir / "PP-LCNet_x1_0_doc_ori.onnx")
        )
    return _ocr


def preprocess_for_ocr(image_np):
    """Preprocess crop image before OCR. Currently a no-op."""
    return image_np


def extract_texts_from_ocr_result(ocr_result):
    """Extract recognized strings from RapidOCR result."""
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


def track_video(video_path: Path, crops_video_dir: Path) -> dict[int, list[tuple[int, Path]]]:
    """Track objects in video and save crops to disk."""
    logger.info(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video info: fps={fps:.1f}, total_frames={total_frames}")

    tracker = get_tracker()
    object_crops = defaultdict(list)
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
            xyxy_boxes = boxes.xyxy

            for i in range(len(xyxy_boxes)):
                try:
                    x1, y1, x2, y2 = map(int, xyxy_boxes[i])
                    track_id = boxes.id[i] if boxes.id is not None else None
                    if track_id is None:
                        obj_id = -1
                    else:
                        obj_id = int(track_id.item() if hasattr(track_id, "item") else track_id)

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

                except Exception as e:
                    logger.warning(f"Error processing box at frame {frame_num}: {e}")
                    continue

        frame_num += 1
        if frame_num % 30 == 0:
            logger.info(f"Processed {frame_num}/{total_frames} frames, tracked {len(tracked_objects)} objects")

    cap.release()
    logger.info(f"Tracking complete. Tracked {len(tracked_objects)} unique objects")
    return dict(object_crops)


def recognize_crops(object_crops: dict[int, list[tuple[int, Path]]]) -> list[dict[str, Any]]:
    """Run OCR on object crops."""
    logger.info(f"Starting OCR for {len(object_crops)} objects...")
    ocr = get_ocr()
    results = []
    total_crops = sum(len(crops) for crops in object_crops.values())
    processed = 0

    for obj_id, crops in sorted(object_crops.items()):
        for frame_num, crop_path in crops:
            try:
                img = cv2.imread(str(crop_path))
                if img is None:
                    logger.warning("Failed to read crop image: %s", crop_path)
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pre = preprocess_for_ocr(img_rgb)

                ocr_result, elapsed = ocr(img_pre)

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


def save_results_csv(ocr_results: list[dict[str, Any]], output_path: Path) -> None:
    """Save OCR results to CSV."""
    df = pd.DataFrame(ocr_results)
    df = df.sort_values(['object_id', 'frame_num']).reset_index(drop=True)

    df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"Results saved to: {output_path}")

    # Print summary
    logger.info(f"Summary: {len(df)} rows, {df['object_id'].nunique()} unique objects, "
                f"{(df['text'].str.len() > 0).sum()} non-empty texts")


async def run_pipeline(video_path: Path, csv_path: Path) -> None:
    """
    Complete pipeline: track video, recognize text, save to CSV.
    Runs synchronously but wrapped as async for compatibility with Celery.
    """
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Create crops directory (storage/crops/<video_name>)
    crops_dir = csv_path.parent / "crops" / video_path.stem
    crops_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Pipeline start: {video_path.name}")
    logger.info("=" * 60)

    try:
        # Step 1: Track objects and collect crops
        logger.info("\n[Step 1/3] Tracking objects in video...")
        object_crops = track_video(video_path, crops_dir)

        if not object_crops:
            raise RuntimeError("No objects tracked in video")

        # Step 2: Sample crops (stride) and run OCR
        logger.info("\n[Step 2/3] Sampling crops with stride=%d and running OCR...", STRIDE)
        sampled = {}
        for obj_id, crops in object_crops.items():
            sampled[obj_id] = [c for idx, c in enumerate(crops) if idx % STRIDE == 0]
        total_before = sum(len(v) for v in object_crops.values())
        total_after = sum(len(v) for v in sampled.values())
        logger.info("Crops: before=%d, after_sampling=%d", total_before, total_after)
        ocr_results = recognize_crops(sampled)

        # Step 3: Save to CSV
        logger.info("\n[Step 3/3] Saving results to CSV...")
        save_results_csv(ocr_results, csv_path)

        logger.info("\n" + "=" * 60)
        logger.info("Pipeline complete!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise
