from ultralytics import YOLO

for m in ("n", "s", "m", "l", "x"):
    # Download the model
    YOLO(f"yolo11{m}.pt")
