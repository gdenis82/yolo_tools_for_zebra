"""
Обучение модели YOLO
"""

if __name__ == '__main__':
    from ultralytics import YOLO, checks
    import torch
    import torchvision
    import os

    checks()

    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

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



    # Load a model
    model = YOLO("yolo11x.pt")

    datasets = os.path.expanduser("split_dataset/dataset.yaml")

    results = model.train(
        data=datasets, epochs=40, imgsz=640, batch=6, device=DEVICE, optimizer='SGD',
        project="ZEBRA", name="train_yolo11x", patience=5, cos_lr=True, close_mosaic=5
    )


    """1
    results = model.train(
        data=datasets, epochs=20, imgsz=640, batch=6, device=DEVICE,
        project="ZEBRA", name="train_yolo11x", patience=5, cos_lr=False
    )
    """

    """2
    results = model.train(
        data=datasets, epochs=20, imgsz=640, batch=6, device=DEVICE,
        project="ZEBRA", name="train_yolo11x", patience=5, cos_lr=True
    )
    """
    """3
    results = model.train(
        data=datasets, epochs=40, imgsz=640, batch=6, device=DEVICE,
        project="ZEBRA", name="train_yolo11x", patience=5, cos_lr=True, close_mosaic=5
    )
    """

    """4
    results = model.train(
        data=datasets, epochs=40, imgsz=640, batch=6, device=DEVICE, optimizer='SGD',
        project="ZEBRA", name="train_yolo11x", patience=5, cos_lr=True, close_mosaic=5
    )
    """