
"""
Описание: Этот скрипт предназначен для балансировки классов аннотаций в YOLO датасете путем копирования объектов
    из недостаточно представленных классов на пустые изображения.

Использование: python balance_classes.py --dataset path_to_dataset --empty path_to_empty_img --output output_folder
param:
 - dataset: Путь к датасету.
 - empty: Путь к пустым изображениям для вставки объектов.
 - output: Папка для сохранения результата балансировки.
"""

import os
import cv2
import argparse
import numpy as np
import random
import uuid

from pathlib import Path
from collections import Counter, defaultdict
from utils import IMAGE_FORMATS, setting_logs

LOG_FILE = 'balance_classes.log'
logging = setting_logs(LOG_FILE)


class YoloDatasetBalancer:
    """
    Класс для балансировки классов аннотаций в YOLO датасете путем копирования объектов
    из недостаточно представленных классов на пустые изображения.
    """

    def __init__(self, dataset_path, empty_images_path, output_path):
        """
        Инициализация балансировщика датасета.
        """
        self.dataset_path = os.path.abspath(dataset_path)
        self.empty_images_path = os.path.abspath(empty_images_path)
        self.output_path = os.path.abspath(output_path)

        self.train_stats = {'class_distribution': Counter()}
        self.valid_stats = {'class_distribution': Counter()}

        self.train_objects_by_class = defaultdict(list)
        self.valid_objects_by_class = defaultdict(list)

        self.empty_images = []

    def analyze_dataset(self):
        """
        Анализ датасета: подсчет распределения классов.
        """
        self._analyze_subset('train')
        self._analyze_subset('valid')
        self._load_empty_images()

        self._print_statistics("До балансировки")

    def _analyze_subset(self, subset):
        """
        Анализ подмножества датасета (train или valid).
        """
        images_dir = os.path.join(self.dataset_path, subset, 'images')
        labels_dir = os.path.join(self.dataset_path, subset, 'labels')

        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            return

        image_files = [f for f in os.listdir(images_dir) if f.endswith(IMAGE_FORMATS)]

        objects_by_class = self.train_objects_by_class if subset == 'train' else self.valid_objects_by_class
        stats = self.train_stats if subset == 'train' else self.valid_stats

        # Анализ аннотаций
        for img_file in image_files:
            label_file = os.path.join(labels_dir, f"{Path(img_file).stem}.txt")

            if not os.path.exists(label_file):
                continue

            with open(label_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                data = line.strip().split()
                if len(data) < 5:
                    continue

                class_id = int(data[0])
                stats['class_distribution'][class_id] += 1

                # Сохраняем объект для последующего использования
                objects_by_class[class_id].append({
                    'image_path': os.path.join(images_dir, img_file),
                    'annotation': line.strip(),
                    'points': [float(x) for x in data[1:]]
                })

    def _load_empty_images(self):
        """
        Загрузка пустых изображений для вставки объектов.
        """
        self.empty_images = [
            os.path.join(self.empty_images_path, f)
            for f in os.listdir(self.empty_images_path)
            if f.endswith(IMAGE_FORMATS)
        ]

    def balance_classes(self):
        """
        Балансировка классов путем копирования объектов на пустые изображения.
        """
        self._prepare_output_directory()
        self._balance_subset('train')
        self._balance_subset('valid')

        self._print_statistics("После балансировки")

    def _prepare_output_directory(self):
        """
        Создание структуры директорий для сбалансированного датасета.
        """
        for subset in ['train', 'valid']:
            images_dir = os.path.join(self.output_path, subset, 'images')
            labels_dir = os.path.join(self.output_path, subset, 'labels')
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)

    def _balance_subset(self, subset):
        """
        Балансировка подмножества датасета.
        """
        stats = self.train_stats if subset == 'train' else self.valid_stats
        objects_by_class = self.train_objects_by_class if subset == 'train' else self.valid_objects_by_class

        if not stats['class_distribution'].values():
            return

        max_annotations = max(stats['class_distribution'].values())
        annotations_added_per_class = defaultdict(int)

        for class_id, count in stats['class_distribution'].items():
            annotations_to_add = max_annotations - count
            if annotations_to_add <= 0:
                continue

            class_objects = objects_by_class[class_id]
            annotations_added = 0
            while annotations_added < annotations_to_add:
                if not self.empty_images:
                    break

                empty_img_path = random.choice(self.empty_images)
                objects_to_add = [random.choice(class_objects)]
                self._copy_multiple_objects_to_empty_image(objects_to_add, empty_img_path, subset)
                annotations_added += 1
                annotations_added_per_class[class_id] += 1

        # Обновление статистики после балансировки
        stats['class_distribution'] = Counter({
            class_id: count + annotations_added_per_class[class_id]
            for class_id, count in stats['class_distribution'].items()
        })

        logging.info(f"Добавлено аннотаций для каждого класса в {subset}: {dict(annotations_added_per_class)}")

    def _copy_multiple_objects_to_empty_image(self, objects, empty_img_path, subset):
        """
        Копирование объектов на пустое изображение.
        """
        empty_img = cv2.imread(empty_img_path)
        if empty_img is None:
            logging.error(f"Не удалось загрузить пустое изображение: {empty_img_path}")
            raise Exception(f"Не удалось загрузить пустое изображение: {empty_img_path}")

        result_img = empty_img.copy()

        all_annotations = []
        for obj in objects:
            src_img = cv2.imread(obj['image_path'])
            if src_img is None:
                continue  # Пропускаем текущий объект, если изображение не удалось загрузить

            # Извлечение аннотации и координат
            data = obj['annotation'].split()
            class_id = int(data[0])
            points = obj['points']  # [x_center, y_center, width, height] в нормализованной форме

            img_h, img_w = src_img.shape[:2]

            # Преобразуем нормализованные координаты в абсолютные пиксельные значения
            x_center, y_center, width_norm, height_norm = points
            w_abs = int(width_norm * img_w)  # Абсолютная ширина объекта
            h_abs = int(height_norm * img_h)  # Абсолютная высота объекта

            x_min = int((x_center * img_w) - (w_abs / 2))
            y_min = int((y_center * img_h) - (h_abs / 2))
            x_max = x_min + w_abs
            y_max = y_min + h_abs

            # Проверка, находятся ли координаты в пределах изображения
            if x_min < 0 or y_min < 0 or x_max > img_w or y_max > img_h:
                continue  # Пропускаем объект, если он выходит за границы

            # Определяем контур объекта (для маски)
            contour_points = [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max]
            ]
            contour = np.array(contour_points, dtype=np.int32)

            # Создаем маску объекта
            mask = np.zeros(src_img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)

            # Извлекаем объект из исходного изображения по маске
            object_roi = cv2.bitwise_and(src_img, src_img, mask=mask)
            object_cropped = object_roi[y_min:y_max, x_min:x_max]

            # Проверяем, помещается ли объект на пустое изображение
            max_x = result_img.shape[1] - w_abs
            max_y = result_img.shape[0] - h_abs

            if max_x <= 0 or max_y <= 0:
                continue  # Пропускаем объект, если он не умещается на пустом изображении

            # Генерируем случайное положение для вставки объекта
            paste_x = random.randint(0, max_x)
            paste_y = random.randint(0, max_y)

            # Вставляем объект в пустое изображение
            paste_region = result_img[paste_y:paste_y + h_abs, paste_x:paste_x + w_abs]
            non_black_mask = object_cropped > 0
            paste_region[non_black_mask] = object_cropped[non_black_mask]

            # Пересчитываем аннотацию для вставленного объекта
            new_x_center = (paste_x + w_abs / 2) / result_img.shape[1]
            new_y_center = (paste_y + h_abs / 2) / result_img.shape[0]
            new_width = w_abs / result_img.shape[1]
            new_height = h_abs / result_img.shape[0]

            # Формируем обновленную аннотацию
            new_annotation = f"{class_id} {new_x_center:.6f} {new_y_center:.6f} {new_width:.6f} {new_height:.6f}"
            all_annotations.append(new_annotation)

        # Сохраняем обновленное изображение и аннотации
        new_img_name = f"{uuid.uuid4()}.jpg"
        new_label_name = f"{Path(new_img_name).stem}.txt"

        cv2.imwrite(os.path.join(self.output_path, subset, 'images', new_img_name), result_img)

        with open(os.path.join(self.output_path, subset, 'labels', new_label_name), 'w') as f:
            for annotation in all_annotations:
                f.write(annotation + '\n')

    def _print_statistics(self, message):
        """
        Вывод статистики распределения классов.
        """
        logging.info(f"\n{message}:")
        logging.info("Распределение классов в тренировочном наборе:")
        for class_id, count in sorted(self.train_stats['class_distribution'].items()):
            logging.info(f"  - Класс {class_id}: {count} аннотаций")

        logging.info("\nРаспределение классов в валидационном наборе:")
        for class_id, count in sorted(self.valid_stats['class_distribution'].items()):
            logging.info(f"  - Класс {class_id}: {count} аннотаций")


def main():
    """
    Основная функция запуска балансировки.
    """
    parser = argparse.ArgumentParser(description='Балансировка классов датасета.')
    parser.add_argument('--dataset', type=str, help='Путь к датасету.')
    parser.add_argument('--empty', type=str, help='Путь к пустым изображениям для вставки объектов.')
    parser.add_argument('--output', type=str, default="balanced_data", help='Папка для сохранения результата балансировки.')

    args = parser.parse_args()

    dataset_path = args.dataset
    empty_images_path = args.empty
    output_path = args.output

    logging.info("Running YOLO dataset balancer...")
    logging.info(f"Params: ")
    logging.info(f"dataset={dataset_path}")
    logging.info(f"empty={empty_images_path}")
    logging.info(f"output={output_path}")

    balancer = YoloDatasetBalancer(dataset_path, empty_images_path, output_path)
    balancer.analyze_dataset()
    balancer.balance_classes()


if __name__ == "__main__":
    main()