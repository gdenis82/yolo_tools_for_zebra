# YOLO Tools for Zebra

## Description
Этот репозиторий содержит набор служебных скриптов для работы с моделями обнаружения и классификации объектов YOLO (You Only Look Once).  
Эти инструменты упрощают подготовку, расширение, преобразование, обучение и проверку наборов данных для задач компьютерного зрения. 
Предназначены для работы с Python 3.9+.

Проект включает в себя полный цикл работы с данными:

1. **Подготовка данных**:
   - Извлечение кадров из видео
   - Аугментация данных
   - Балансировка классов
   - Разделение на тренировочный и тестовый наборы

2. **Обучение моделей**:
   - Использование предобученной модели (yolo11x.pt)

3. **Оценка и визуализация**:
   - Валидация метрик модели
   - Визуализация аннотаций
   - Создание графиков и матриц ошибок

4. **Применение**:
   - Предсказание по видео
   - Детекция объектов

Проект организован таким образом, чтобы обеспечить четкое разделение между различными этапами обработки данных и обучения моделей, 
с сохранением промежуточных результатов в соответствующих директориях.

## Prerequisites
Прежде чем начать, убедитесь, что у вас установлены следующие предварительные требования: 

Step 0. Скачайте и установите Miniconda с официального [website](https://docs.anaconda.com/miniconda/).

Step 1. Создайте новую среду conda с Python версии 3.10 или выше и активируйте ее.

```
conda create --name zebraTest python=3.10 -y
```
```
conda activate zebraTest
```

или используйте venv
```
python3.10 -m venv .venv
```
```
windows:
.\.venv\Scripts\activate

linux:
source .venv/bin/activate
```

Step 3. Скачайте и установите FFmpeg.

Чтобы установить FFmpeg, выполните следующие действия:
1. Посетите официальный сайт FFmpeg: [ffmpeg.org](https://www.ffmpeg.org/download.html).
2. Загрузите соответствующий пакет.
3. Следуйте инструкциям по установке для вашей платформы.

## Installation

1. Клонировать этот репозиторий
2. Установите необходимые зависимости:

```
pip install -r requirements.txt
```

Для обучения на GPU
```
pip uninstall torch torchvision
```
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
Включить ведение журнала на TensorBoard
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
7. После проверки!!! Удалить промежуточные папки вывода, оставив конечный результат. 
Промежуточные данные позволяют проверять результат и в случаях не стабильной работы/добавления дополнительных опций, позволяют вернуться на шаг назад.

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
- matplotlib (visualization)
- scikit-learn (for dataset splitting)
- tqdm (progress bars)
- ffmpeg-python (video processing)
- albumentations (data augmentation)
- tensorboard (logging and visualization)

## Основные файлы

```
ZebraTest/
├── augment_dataset.py          # Скрипт для аугментации данных
├── balance_classes.py          # Скрипт для балансировки классов
├── convert_ann.py              # Скрипт конвертации .json аннотаций в yolo фармат
├── extract_frames.py           # Извлечение кадров из видео
├── prediction_by_video.py      # Предсказание по видео
├── README.md                   # Документация проекта
├── requirements.txt            # Зависимости проекта
├── split_dataset.py            # Разделение датасета
├── REPORT.md                   # Отчет о выполнении задания
├── train.py                    # Скрипт для обучения модели
├── utils.py                    # Вспомогательные функции
├── valid_model_metrics.py      # Валидация метрик модели
├── visualize_annotations.py    # Визуализация аннотаций
└
```





