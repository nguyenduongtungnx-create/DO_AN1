"""
import cv2
import torch
from ultralytics import YOLO

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8s.pt")
TRAFFIC_CLASSES = [0, 1, 2, 3, 5, 7]

results = model.track(
    source="videos/traffic.mp4",
    conf=0.10,
    iou=0.45,
    imgsz=960,
    classes=TRAFFIC_CLASSES,
    tracker="bytetrack.yaml",
    persist=True,
    device=device,
    stream=True,
    show=False,  # Tắt tính năng tự hiện của YOLO để tự quản lý bằng OpenCV
    verbose=False
)

# Thay vòng lặp 'pass' bằng vòng lặp hiển thị của OpenCV
for r in results:
    # Lấy khung hình đã được YOLO vẽ sẵn khung (bounding box)
    annotated_frame = r.plot()

    # Hiển thị cửa sổ video
    cv2.imshow("Tracking Video", annotated_frame)

    # Thêm hàm waitKey để cửa sổ không bị đơ và nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Đóng cửa sổ khi kết thúc
cv2.destroyAllWindows()
"""
import time
import cv2
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
MODEL_PATH = "weights/yolov8s.pt"
VIDEO_PATH = "videos/traffic.mp4"

# COCO classes: 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASSES = [2, 3, 5, 7]

CONF = 0.25
IOU = 0.45
IMG_SIZE = 640

# =========================
# LOAD MODEL & VIDEO
# =========================
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"LỖI: Không thể mở video tại '{VIDEO_PATH}'!")
    raise SystemExit

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps_input = cap.get(cv2.CAP_PROP_FPS)

window_name = "YOLOv8 Traffic Tracking"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

frame_count = 0
vehicle_ids = set()
prev_time = time.time()

# =========================
# PROCESS VIDEO
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Run ByteTrack tracker
    results = model.track(
        frame,
        conf=CONF,
        iou=IOU,
        imgsz=IMG_SIZE,
        classes=VEHICLE_CLASSES,
        tracker="bytetrack.yaml",
        persist=True,
        verbose=True  # Đã bật True để in log chi tiết ra terminal
    )

    r = results[0]

    # Update tracked IDs
    if r.boxes is not None and r.boxes.id is not None:
        ids = r.boxes.id.int().tolist()
        vehicle_ids.update(ids)

    # Annotate frame (vẽ Bounding Box & ID lên ảnh)
    annotated = r.plot()

    # Tính toán FPS thực tế
    curr_time = time.time()
    fps_real = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    # Hiển thị thông số trên Frame
    cv2.putText(
        annotated,
        f"Frame: {frame_count}/{total_frames}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        annotated,
        f"Unique IDs Tracker: {len(vehicle_ids)}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        annotated,
        f"FPS: {fps_real:.1f}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    cv2.imshow(window_name, annotated)

    # Bấm Q để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =========================
# CLEAN UP
# =========================
cap.release()
cv2.destroyAllWindows()

print("=" * 40)
print("Frames processed :", frame_count)
print("Unique track IDs :", len(vehicle_ids))
print("=" * 40)