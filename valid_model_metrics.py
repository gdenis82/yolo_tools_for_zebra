
if __name__ == '__main__':
    from ultralytics import YOLO
    import torch
    import torchvision
    import os

    DEVICE = []
    print(f"PyTorch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        print(f"Number of devices: {num_devices}")

        for i in range(num_devices):
            DEVICE.append(i)
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No CUDA-compatible devices found. Using CPU.")

    # Загрузка модели
    model = YOLO("ZEBRA/train_yolo11x4/weights/best.pt")

    # Оценка модели
    datasets = os.path.expanduser("split_dataset/dataset.yaml")
    metrics = model.val(data=datasets, imgsz=640, batch=6, device=DEVICE,
        project="ZEBRA", name="val_yolo11x",)

    if metrics.task == 'detect':
        # Метрики точности (для детекций)
        print("mAP50-95 (Box Detection): Средняя точность на IoU от 0.5 до 0.95")
        print("mAP50-95: ", metrics.box.map)
        print("mAP50 (Box Detection): Средняя точность на IoU = 0.5")
        print("mAP50: ", metrics.box.map50)
        print("mAP75 (Box Detection): Средняя точность на IoU = 0.75")
        print("mAP75: ", metrics.box.map75)
        print("mAP50-95 per class (Box Detection): ", metrics.box.maps)

        # Метрики полноты и точности (по классам или усреднённые можно взять через mean_results)
        box_results = metrics.box.mean_results()  # Получить усреднённые результаты для детекций
        print("Mean Box Precision (mp): ", box_results[0])
        print("Mean Box Recall (mr): ", box_results[1])

        # Метрики скорости
        print("Inference time (ms): Время на обработку одного кадра")
        print("Inference time (ms): ", metrics.speed['inference'])  # Время на обработку
        print("Frames per second (FPS): Количество кадров, обрабатываемых за секунду")
        print("FPS: ", 1000 / metrics.speed['inference'])  # FPS на основе времени вывода