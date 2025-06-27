
"""
Описание: Этот скрипт разбивает набор данных на train, valid, test наборы с подсчетом аннотаций по классам.

Использование:
python split_dataset.py --dataset /path/to/dataset --output /path/to/output --train_ratio 0.7 --val_ratio 0.2 --test_ratio 0.1

param:
 - dataset: Путь к исходному набору данных
 - output: Путь для сохранения результата (train, val, test)
 - train_ratio: Пропорция для train набора данных (по умолчанию 70%)
 - val_ratio: Пропорция для val набора данных (по умолчанию 20%)
 - test_ratio: Пропорция для test набора данных (по умолчанию 10%)
"""

import os
import random
import shutil
import argparse

from pathlib import Path
from sklearn.model_selection import train_test_split
from utils import count_annotations, setting_logs

LOG_FILE = 'split_dataset.log'
logging = setting_logs(LOG_FILE)


def move_files(image_list, source_dir, target_dir):
    """
    Функция для перемещения изображений и аннотаций в соответствующие директории
    :param image_list:
    :param source_dir:
    :param target_dir:
    :return:
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    images_path = os.path.join(source_dir, "images")
    labels_path = os.path.join(source_dir, "labels")

    target_dir_images = os.path.join(target_dir, "images")
    target_dir_labels = os.path.join(target_dir, "labels")

    os.makedirs(target_dir_images, exist_ok=True)
    os.makedirs(target_dir_labels, exist_ok=True)

    for image in image_list:
        # Перемещаем изображение
        shutil.copy(os.path.join(images_path, image), target_dir_images)

        # Перемещаем соответствующий файл аннотации .txt
        annotation_file = f"{Path(image).stem}.txt"
        annotation_path = os.path.join(labels_path, annotation_file)

        if os.path.exists(annotation_path):
            shutil.copy(annotation_path, target_dir_labels)

def print_statistics(dataset_name, dataset_dir):
    """
    Функция для вывода статистики по набору данных
    :param dataset_name:
    :param dataset_dir:
    :return:
    """
    image_files, class_counts, total_annotations = count_annotations(dataset_dir)
    logging.info(f"Статистика для набора {dataset_name}:")
    logging.info(f"Количество изображений: {len(image_files)}")
    logging.info(f"Общее количество аннотаций: {total_annotations}")

    # Подсчитываем количество аннотаций по каждому классу
    logging.info("Аннотации по классам:")
    for class_id, count in class_counts.items():
        logging.info(f"Класс {class_id}: {count} аннотаций")


def main():
    """
    Главная функция для обработки аргументов и выполнения разделения данных
    """
    parser = argparse.ArgumentParser(
        description="Разделить набор данных на train, val, и test с подсчетом аннотаций по классам.")
    parser.add_argument('--dataset', type=str, required=True, help="Путь к исходному набору данных")
    parser.add_argument('--output', type=str, required=True,
                        help="Путь для сохранения результата (train, val, test)")
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help="Пропорция для train набора данных (по умолчанию 70%)")
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help="Пропорция для val набора данных (по умолчанию 20%)")
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help="Пропорция для test набора данных (по умолчанию 10%)")

    # Чтение аргументов
    args = parser.parse_args()
    dataset_path = args.dataset
    output_dir = args.output
    test_ratio = args.test_ratio
    val_ratio = args.val_ratio
    train_ratio = args.train_ratio

    logging.info("Running split dataset script...")
    logging.info(f"Params: ")
    logging.info(f"dataset= {dataset_path}")
    logging.info(f"output= {output_dir}")
    logging.info(f"train_ratio= {train_ratio}")
    logging.info(f"val_ratio= {val_ratio}")
    logging.info(f"test_ratio= {test_ratio}")

    images_path = os.path.join(dataset_path, "images")

    # Получаем все файлы с расширением .jpg (или другие нужные файлы)
    image_files = [f for f in os.listdir(images_path) if f.endswith('.jpg')]

    # Проверка на пустую директорию
    if not image_files:
        logging.error(f"В указанной директории {images_path} не найдено изображений.")
        raise ValueError(f"В указанной директории {images_path} не найдено изображений.")

    # Разделяем данные на train и temp (train + val + test)
    train_files, temp_files = train_test_split(image_files, train_size=train_ratio, random_state=42)

    # Разделяем temp на val и test
    val_files, test_files = train_test_split(temp_files, test_size=test_ratio / (val_ratio + test_ratio),
                                             random_state=42)

    # Создаем папки для train, val и test
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')

    # Перемещаем изображения и аннотации в соответствующие директории
    move_files(train_files, dataset_path, train_dir)
    move_files(val_files, dataset_path, val_dir)
    move_files(test_files, dataset_path, test_dir)

    logging.info(f"Train: {len(train_files)} images and annotations")
    logging.info(f"Validation: {len(val_files)} images and annotations")
    logging.info(f"Test: {len(test_files)} images and annotations")

    # Выводим статистику для каждого набора данных
    print_statistics("Train", train_dir)
    print_statistics("Validation", val_dir)
    print_statistics("Test", test_dir)


if __name__ == "__main__":
    main()