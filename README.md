# YOLO Tools for Zebra

Этот репозиторий содержит набор служебных скриптов для работы с моделями обнаружения и классификации объектов YOLO (You Only Look Once).  
Эти инструменты упрощают подготовку, расширение, преобразование, обучение и проверку наборов данных для задач компьютерного зрения. 
Предназначены для работы с Python 3.9+ в Windows.

## Prerequisites
Before you start, ensure that you have the following prerequisites installed:

Step 0. Download and install Miniconda from the official [website](https://docs.anaconda.com/miniconda/).

Step 1. Create a new conda environment with Python version 3.10 or higher, and activate it.

```
conda create --name zebraTest python=3.10 -y
```
```
conda activate zebraTest
```

or use venv
```
python3.10 -m venv .venv
```
```
windows:
.\.venv\Scripts\activate

linux:
source .venv/bin/activate
```

Step 3. Download and install FFmpeg.

To install FFmpeg, follow these steps:
1. Visit the official FFmpeg website: [ffmpeg.org](https://www.ffmpeg.org/download.html).
2. Download the appropriate package.
3. Follow the installation instructions for your platform.

## Installation

1. Clone this repository
2. Install the required dependencies:

```
pip install -r requirements.txt
```

For train GPU
```
pip uninstall torch torchvision
```
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
Enable TensorBoard logging
```
yolo settings tensorboard=True
```
## Tools Overview

### Dataset Preparation

#### [extract_frames.py](#extract_framespy)

Предназначен для извлечения кадров из видеофайлов (по умолчанию 1 кадр в секунду).

Usage: 
```
python extract_frames.py --video path_to_video --output output_folder
```
param:
 - video: Путь к видео файлам или архиву (zip, архив будет распакован в текущую папку расположения архива)
 - output: Папка для сохранения кадров.

#### [balance_classes.py](#balance_classespy)
Балансирует распределение классов в наборах данных YOLO, анализируя частоту классов и копируя объекты на пустые 
изображения для получения более сбалансированных обучающих данных.

Usage:
```
python balance_classes.py --dataset path_to_dataset --empty path_to_empty_img --output output_folder
```
param:
 - dataset: Путь к датасету.
 - empty: Путь к пустым изображениям для вставки объектов.
 - output: Папка для сохранения результата балансировки.

#### [augment_dataset.py](#augment_datasetpy)
Инструмент для расширения наборов данных YOLO с помощью таких преобразований, как переворот и поворот. 
 Поддерживает как ограничивающие рамки, так и аннотации сегментации (опционально).

Usage:
```
python augment_dataset.py --dataset /path/to/dataset --output /path/to/output --mode bboxes --debug_dir /path/to/output_debug
```
param:
 - dataset: Путь к корневой папке датасета (содержит train, valid, test папки).
 - output: Путь для записи нового аугментированного датасета.
 - mode: Тип датасета. "bboxes" — обработка боксов, "contours" — обработка контуров.
 - debug_dir: Папка для сохранения изображений с аннотациями (опционально).

#### [split_dataset.py](#split_datasetpy)
Разбивает набор данных на train, valid, test наборы с подсчетом аннотаций по классам.

Usage:
```
python split_dataset.py --dataset /path/to/dataset --output /path/to/output --train_ratio 0.7 --val_ratio 0.2 --test_ratio 0.1
```

param:
 - dataset: Путь к исходному набору данных
 - output: Путь для сохранения результата (train, val, test)
 - train_ratio: Пропорция для train набора данных (по умолчанию 70%)
 - val_ratio: Пропорция для val набора данных (по умолчанию 20%)
 - test_ratio: Пропорция для test набора данных (по умолчанию 10%)

### Debugging and Visualization

#### `visualize_annotations.py`
Визуализирует аннотации YOLO, рисуя прямоугольники на изображениях. Полезно для проверки правильности аннотаций.
 Проверяет структуру папок датасета. Отображает статистику по набору данных.

### Training and Validation

#### [train.py](#trainpy)
Скрипт для обучения модели YOLO с использованием библиотеки Ultralytics. 
Проверяет наличие CUDA, загружает предварительно обученную модель и обучает ее на указанном наборе данных.
Измените параметры обучения согласно Вашим требованиям. Подробная информация в документации на сайте [Ultralytics](https://docs.ultralytics.com/ru/modes/train/)

#### [valid_model_metrics.py](#valid_model_metricspy)
Оценивает обученную модель YOLO на наборе данных для проверки и выводит различные показатели, 
включая mAP (среднюю точность), точность, полноту и скорость вывода.

## Usage Examples

В верхней части каждого скрипта приведен пример его использования. 
Чтобы использовать скрипт, измените пути ввода/вывода и параметры по мере необходимости, а затем запустите:

```
python script_exemple_name.py
```

Для большинства скриптов вам потребуется указать: 
 - входной каталог, содержащий изображения и/или аннотации,
 - выходной каталог для обработанных файлов,
 - любые дополнительные параметры для задачи (например, путь к папке изображений без аннотаций/объектов для балансировки см. примеры в описание скриптов)


#### Последовательность подготовки данных

1. Скачать видео: [Хранилище на Яндекс.Диске](https://disk.yandex.ru/d/-VhiX2BOWdw-rg)
2. Извлечь кадры: [extract_frames.py](#extract_framespy)
3. Аннотировать изображения, экспорт данных в структуре (train/images/img1.jpg, train/labels/img1.txt) (bounding boxes + классы): Roboflow, Cvat, X-AnyLabeling, ... .
4. Балансировать классы аннотаций (опционально, если требуется): [balance_classes.py](#balance_classespy)
5. Провести аугментацию данных (переворот, поворот. Всего: *8): [augment_dataset.py](#augment_datasetpy)
6. Разбить на train/val/test (70/20/10):[split_dataset.py](#split_datasetpy)
7. После проверки!!! Очистить проект подготовки данных. Удалить промежуточные папки вывода, оставив конечный результат. 
Промежуточные данные позволяют проверять результат и в случаях не стабильной работы/добавления дополнительных опций позволяют вернуться на шаг назад (и не начинать с начала), 
после правок в скрипте подготовки данных.

#### Структура датасета:
```
Dataset
├── dataset.yaml
├── train
│   ├── images
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── img3.jpg
│   └── labels
│       ├── img1.txt
│       ├── img2.txt
│       └── img3.txt
├── val
│   ├── images
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── img3.jpg
│   └── labels
│       ├── img1.txt
│       ├── img2.txt
│       └── img3.txt
└── test
    ├── images
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── img3.jpg
    └── labels
        ├── img1.txt
        ├── img2.txt
        └── img3.txt


```
#### Файл конфигурации датасета dataset.yaml:
```
train: ./train/images
val: ./val/images
test: ./test/images

nc: 3
names:
  0: class1
  1: class2
  2: class3
```
## Requirements

Инструменты в этом репозитории зависят от различных библиотек Python, включая:
- ultralytics (YOLO implementation)
- opencv-python (image processing)
- numpy (numerical operations)
- matplotlib (visualization)
- PIL/Pillow (image manipulation)
- scikit-learn (for dataset splitting)
- tqdm (progress bars)
- ffmpeg-python (video processing)
- albumentations (data augmentation)
- tensorboard (logging and visualization)





