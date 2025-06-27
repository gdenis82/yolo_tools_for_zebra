
"""
Описание: Этот скрипт проверяет наличия в train, val, test папок images, labels.
Визуализирует рандомные изображения и их аннотации из датасета.

Использование:
python visualize_annotations.py --dataset path/to/dataset --num_samples 5

param:
 - dataset: Путь к исходному набору данных
 - num_samples: Число отображаемых образцов (по умолчанию 5)
"""

import os
import argparse
import yaml
import random

from pathlib import Path
from ultralytics.data.utils import visualize_image_annotations

from utils import count_annotations, setting_logs, SUBSET_NAMES

LOG_FILE = 'visualize_annotations.log'
logging = setting_logs(LOG_FILE)

def check_dataset_structure(dataset_path):
    """
    Функция для проверки наличия в train, val, test папок images, labels в датасете.
    Отображение статистики датасета.
    :param dataset_path:
    :return:
    """
    missing_folders = []

    for subset in SUBSET_NAMES:
        subset_path = os.path.join(dataset_path, subset)

        if os.path.exists(subset_path):
            for folder in ('images', 'labels'):

                folder_path = os.path.join(subset_path, folder)

                if not os.path.exists(folder_path):
                    missing_folders.append(folder_path)

            print_statistics(subset, dataset_path)
        else:
            missing_folders.append(subset_path)


    if missing_folders:
        print(f"Отсутствуют необходимые папки: {', '.join(missing_folders)}")
        return False
    return True


def load_class_names(yaml_path):
    """
    Функция для извлечения names классов из YAML файла
    :param yaml_path:
    :return:
    """
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)

    return data['names']


def visualize_annotations(image_path, label_path, labels):
    """
    Функция визуализации
    :param image_path:
    :param label_path:
    :param labels:
    :return:
    """

    visualize_image_annotations(
        image_path=image_path,
        txt_path=label_path,
        label_map=labels,
    )

def print_statistics(subset, dataset_dir):
    """
    Функция для вывода статистики по набору данных
    :param subset:
    :param dataset_dir:
    :return:
    """
    image_files, class_counts, total_annotations = count_annotations(os.path.join(dataset_dir, subset))
    logging.info(f"Статистика для набора {subset}:")
    logging.info(f"Количество изображений: {len(image_files)}")
    logging.info(f"Общее количество аннотаций: {total_annotations}")

    # Подсчитываем количество аннотаций по каждому классу
    logging.info("Аннотации по классам:")
    for class_id, count in class_counts.items():
        logging.info(f"Класс {class_id}: {count} аннотаций")

def main():
    # Парсер аргументов
    parser = argparse.ArgumentParser(description="Визуализация изображений для YOLO.")

    # Добавление аргументов
    parser.add_argument("--dataset", type=str, help="Путь к датасету")
    parser.add_argument("--num_samples", type=int, default=5, help="Количество отображаемых образцов (По умолчанию 5)")

    # Разбор аргументов
    args = parser.parse_args()
    dataset_dir = args.dataset
    num_samples = args.num_samples

    # Проверка структуры датасета
    if not check_dataset_structure(dataset_dir):
        logging.error("Структура датасета неверна")
        raise Exception("Структура датасета неверна")

    # Путь к YAML файлу с классами
    yaml_file_path = os.path.join(dataset_dir, 'dataset.yaml')

    if not os.path.exists(yaml_file_path):
        print(f"Не найден файл {yaml_file_path}")
        raise Exception(f"Не найден файл {yaml_file_path}")

    # Загрузка имен классов
    label_map = load_class_names(yaml_file_path)

    # Путь к папке с изображениями
    images_folder = os.path.join(dataset_dir, "train/images")

    # Получаем список всех файлов изображений
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Выбираем случайные изображений
    random_images = random.sample(image_files, min(num_samples, len(image_files)))

    # Для каждого выбранного изображения визуализируем аннотации
    for image_file in random_images:
        image_dir = os.path.join(images_folder, image_file)
        label_dir = os.path.join(dataset_dir, "train/labels", f"{Path(image_file).stem}.txt")

        # Вызов функции для визуализации
        visualize_annotations(image_dir, label_dir, label_map)

if __name__ == "__main__":
    main()

    # Опционально, если нужно только получить статистику по набору (предварительно задокументировать #main())
    #check_dataset_structure('balanced_data')