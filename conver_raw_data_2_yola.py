import csv
import argparse
import random
import shutil
import subprocess
import logging
import traceback
from pathlib import Path
from PIL import Image

def to_float(s):
    if s is None or s == "":
        return None
    s = str(s).strip().replace(" ", "")
    s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return None

def extract_frame(video_path: Path, frame_idx: int, out_path: Path):
    if out_path.exists():
        return True
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Резервный вариант OpenCV для получения fps и конвертации
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logging.warning("cv2 не может открыть видео %s", video_path)
            return False
        
        # Получить информацию о видео
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logging.info("📹 Видео %s: кадров=%d fps=%.1f разрешение=%dx%d", 
                     video_path.name, total_frames, fps, width, height)
        
        # Если frame_idx >= total_frames, попробуем интерпретировать как миллисекунды
        original_frame_idx = frame_idx
        if frame_idx >= total_frames:
            logging.debug("⏱️  frame_idx=%d >= total_frames=%d, пытаемся интерпретировать как миллисекунды", frame_idx, total_frames)
            frame_idx_from_ms = int(frame_idx * fps / 1000.0)
            if frame_idx_from_ms < total_frames:
                logging.debug("✅ Преобразовано %d мс → кадр %d", original_frame_idx, frame_idx_from_ms)
                frame_idx = frame_idx_from_ms
            else:
                logging.error("❌ Даже после преобразования: кадр %d >= всего кадров %d в видео %s", frame_idx_from_ms, total_frames, video_path.name)
                cap.release()
                return False
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            logging.error("❌ cv2 не смог прочитать кадр %d из видео %s (всего %d кадров)", 
                         frame_idx, video_path.name, total_frames)
            return False
        # write image using cv2
        try:
            cv2.imwrite(str(out_path), frame)
            logging.debug("✅ Кадр %d извлечен в %s", frame_idx, out_path.name)
            return out_path.exists()
        except Exception as e:
            logging.error("❌ cv2 ошибка записи изображения %s: %s\n%s", 
                         out_path, e, traceback.format_exc())
            return False
    except Exception as e:
        logging.error("❌ Ошибка cv2: %s\n%s", e, traceback.format_exc())
        return False

def prepare(data_dir: Path, out_images: Path, out_labels: Path, train_ratio: float, class_index: int):
    csv_files = list(data_dir.rglob("*.csv"))
    if not csv_files:
        logging.info("CSV файлы не найдены в %s", data_dir)
        return

    stats = {
        "rows_total": 0,
        "rows_no_filename": 0,
        "rows_src_not_found": 0,
        "rows_no_frame_idx": 0,
        "rows_extraction_failed": 0,
        "rows_no_bbox": 0,
        "rows_invalid_bbox": 0,
        "images_saved": 0
    }

    for csv_path in csv_files:
        logging.info("📄 Обработка %s", csv_path)
        with csv_path.open(encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stats["rows_total"] += 1
                fname = (row.get("filename") or "").strip()
                if not fname:
                    stats["rows_no_filename"] += 1
                    continue

                # determine source: video or image
                src_path = data_dir / fname
                # if filename is not a path under data_dir, try same folder as csv
                if not src_path.exists():
                    src_path = csv_path.parent / fname

                if not src_path.exists():
                    logging.debug("Источник не найден: %s (пробовали %s и %s)", 
                                 fname, data_dir / fname, csv_path.parent / fname)
                    stats["rows_src_not_found"] += 1
                    continue

                frame_ts = to_float(row.get("frame_timestamp"))
                # attempt to treat frame_ts as frame index
                frame_idx = int(frame_ts) if frame_ts is not None else None

                if src_path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv") and frame_idx is None:
                    logging.debug("Видео но нет индекса кадра: %s", fname)
                    stats["rows_no_frame_idx"] += 1
                    continue

                # construct image name: use csv stem + frame_idx to avoid collisions
                img_stem = f"{csv_path.stem}_{frame_idx:06d}" if frame_idx is not None else (Path(fname).stem)
                img_name = img_stem + ".jpg"

                # target split
                split = "train" if random.random() < train_ratio else "val"
                img_out_dir = out_images / split
                lbl_out_dir = out_labels / split
                img_out_dir.mkdir(parents=True, exist_ok=True)
                lbl_out_dir.mkdir(parents=True, exist_ok=True)
                img_out_path = img_out_dir / img_name
                lbl_out_path = lbl_out_dir / (img_out_path.stem + ".txt")

                # if source is video -> extract frame
                if src_path.exists() and src_path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
                    ok = extract_frame(src_path, frame_idx, img_out_path)
                    if not ok:
                        logging.debug("Извлечение кадра не удалось: %s кадр %d", src_path, frame_idx)
                        stats["rows_extraction_failed"] += 1
                        continue
                else:
                    # assume source is already an image file somewhere; try to copy from csv folder or data_dir
                    candidate = csv_path.parent / fname

                    def copy_or_find(src_path: Path):
                        # return True if image written to img_out_path
                        if not src_path.exists():
                            return False
                        if src_path.is_file():
                            try:
                                shutil.copy2(src_path, img_out_path)
                                return True
                            except Exception as e:
                                logging.error("Ошибка копирования %s -> %s: %s\n%s", 
                                            src_path, img_out_path, e, traceback.format_exc())
                                return False
                        if src_path.is_dir():
                            # try to find media file inside directory
                            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".avi", ".mov", ".mkv"):
                                for found in src_path.rglob(f"*{ext}"):
                                    if found.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
                                        if frame_idx is None:
                                            logging.debug("Нет индекса кадра для видео внутри папки %s, пропускаем строку", src_path)
                                            return False
                                        return extract_frame(found, frame_idx, img_out_path)
                                    else:
                                        try:
                                            shutil.copy2(found, img_out_path)
                                            return True
                                        except Exception as e:
                                            logging.error("Ошибка копирования %s -> %s: %s\n%s", 
                                                        found, img_out_path, e, traceback.format_exc())
                                            return False
                            return False
                        return False

                    if copy_or_find(candidate):
                        pass
                    else:
                        candidate2 = data_dir / fname
                        if copy_or_find(candidate2):
                            pass
                        else:
                            logging.debug("Изображение не найдено для %s; пропущено", fname)
                            stats["rows_src_not_found"] += 1
                            continue

                # open image and compute normalized bbox
                try:
                    im = Image.open(img_out_path).convert("RGB")
                except Exception as e:
                    logging.error("Ошибка открытия изображения %s: %s\n%s", 
                                 img_out_path, e, traceback.format_exc())
                    continue
                W, H = im.size

                x_min = to_float(row.get("x_min"))
                y_min = to_float(row.get("y_min"))
                x_max = to_float(row.get("x_max"))
                y_max = to_float(row.get("y_max"))

                if None in (x_min, y_min, x_max, y_max):
                    stats["rows_no_bbox"] += 1
                    # skip if bbox absent
                    continue

                # ensure bbox valid
                x_min, x_max = min(x_min, x_max), max(x_min, x_max)
                y_min, y_max = min(y_min, y_max), max(y_min, y_max)
                cx = (x_min + x_max) / 2.0 / W
                cy = (y_min + y_max) / 2.0 / H
                bw = (x_max - x_min) / W
                bh = (y_max - y_min) / H

                if bw <= 0 or bh <= 0:
                    stats["rows_invalid_bbox"] += 1
                    continue

                # append label (multiple rows for same image will append multiple lines)
                with lbl_out_path.open("a", encoding="utf-8") as lf:
                    lf.write(f"{class_index} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                stats["images_saved"] += 1

    logging.info("Готово. Изображения: %s Ярлыки: %s", out_images, out_labels)
    logging.info("Статистика: всего_строк=%d нет_имени=%d источник_не_найден=%d нет_кадра=%d ошибка_извлечения=%d нет_bbox=%d неверный_bbox=%d => сохранено_изображений=%d",
                 stats["rows_total"], stats["rows_no_filename"], stats["rows_src_not_found"],
                 stats["rows_no_frame_idx"], stats["rows_extraction_failed"], stats["rows_no_bbox"],
                 stats["rows_invalid_bbox"], stats["images_saved"])

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--raw", dest="data_dir", type=Path, default=Path("RawData"), help="папка с CSV и видео/изображениями")
    p.add_argument("--out", dest="out_root", type=Path, default=Path("dataset"), help="выходная папка (создаст images/ и labels/)")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--class-index", type=int, default=0)
    p.add_argument("--seed", type=int, default=42, help="случайное зерно для разделения train/val")
    args = p.parse_args()
    random.seed(args.seed)
    out_images = args.out_root / "images"
    out_labels = args.out_root / "labels"
    prepare(args.data_dir, out_images, out_labels, args.train_ratio, args.class_index)