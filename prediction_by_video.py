from ultralytics import YOLO

model = YOLO("ZEBRA/train_yolo11x4/weights/best.pt")

results = model.predict("raw_data/тест/1.MOV", save=True, show=False)