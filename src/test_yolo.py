from ultralytics import YOLO

# 1. Tải và lưu mô hình vào đúng thư mục weights/
model = YOLO("weights/yolov8n.pt")

# 2. Chạy nhận diện với stream=True
results = model.predict(
    source="videos/traffic.mp4",
    show=True,
    conf=0.4,
    stream=True  # Xử lý luồng từng frame, tối ưu bộ nhớ
)

# 3. Duyệt qua từng frame để hiển thị
for r in results:
    pass