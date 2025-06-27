"""
Вспомогательные функции
"""

import os
import glob
import logging
import zipfile

from pathlib import Path

LOG_FOLDER = 'logs'
IMAGE_FORMATS = ('.jpg', '.jpeg', '.png')
VIDEO_FORMATS = ('.mp4', '.avi', '.mov')
SUBSET_NAMES = ('train', 'val', 'test')

Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)


def setting_logs(log_file)-> logging:
    # Настройка логирования
    log_file = os.path.join(LOG_FOLDER, log_file)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),  # Логи в файл
            logging.StreamHandler()  # Логи в консоль
        ]
    )
    return logging

def count_annotations(dataset_dir):
    """
    Функция для подсчета аннотаций для каждого класса в наборе данных
    :param dataset_dir:
    :return:
    """
    class_counts = {}
    total_annotations = 0
    image_files = [f for f in os.listdir(os.path.join(dataset_dir, "images")) if f.endswith(IMAGE_FORMATS)]

    for image_file in image_files:
        annotation_file = f"{Path(image_file).stem}.txt"
        annotation_path = os.path.join(dataset_dir, "labels",  annotation_file)

        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                lines = f.readlines()
                total_annotations += len(lines)
                for line in lines:
                    class_id = int(line.split()[0])  # Первый элемент — это ID класса
                    if class_id in class_counts:
                        class_counts[class_id] += 1
                    else:
                        class_counts[class_id] = 1
    return image_files, class_counts, total_annotations

def find_all_image_files(directory):
    """
    Возвращает список путей к изображениям в указанной директории и её поддиректориях,
    соответствующим расширениям из SUFFIXES.

    :param directory: Директория для поиска изображений.
    :return: Список путей к файлам изображений.
    """

    image_paths = []
    for suffix in IMAGE_FORMATS:
        image_paths.extend(glob.glob(os.path.join(directory, f"**/*{suffix}"), recursive=True))

    return image_paths

def find_all_video_files(directory):
    """
    Рекурсивный поиск всех видео файлов поддерживаемым форматом в заданной директории и её поддиректориях.
    """
    video_files = []  # Список для хранения найденных файлов

    # Рекурсивный обход всех папок
    for root, _, files in os.walk(directory):
        for file in files:
            # Проверяем, является ли файл поддерживаемым форматом
            if Path(file).suffix.lower() in VIDEO_FORMATS:
                video_files.append(os.path.join(root, file))  # Сохраняем полный путь к файлу

    return video_files

def extract_zip_file(zip_path, current_logging):
    """
    Распаковывает zip-архив в директории архива.

    :param current_logging:
    :param zip_path: Путь к zip-файлу.
    """

    try:

        # Проверяем, существует ли zip-файл
        if not os.path.isfile(zip_path):
            raise FileNotFoundError(f"Файл {zip_path} не найден!")

        extract_to = Path(zip_path).parent

        # Открываем zip-файл и извлекаем его содержимое
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            current_logging.info(f"Архив {zip_path} успешно распакован в {extract_to}.")

    except zipfile.BadZipFile:
        current_logging.error(f"Ошибка: файл {zip_path} не корректный zip-архив.")
    except Exception as e:
        current_logging.error(f"Произошла ошибка: {e}")
