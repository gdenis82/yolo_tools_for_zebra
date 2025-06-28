
"""
Описание: Этот скрипт предназначен для аугментации аннотированных фотографий.

Использование:
python augment_dataset.py --dataset /path/to/dataset --output /path/to/output --mode bboxes --debug_dir /path/to/output_debug

param:
 - dataset: Путь к корневой папке датасета (содержит train, valid, test папки).
 - output: Путь для записи нового аугментированного датасета.
 - mode: Тип датасета. "bboxes" - обработка боксов, "contours" - обработка контуров.
 - debug_dir: Папка для сохранения изображений с аннотациями (опционально).
"""

import os
from pathlib import Path

import cv2
import argparse
import numpy as np

from collections import Counter, defaultdict
from utils import setting_logs, find_all_image_files

LOG_FILE = 'augment_dataset.log'

TRANSFORMATIONS = (
        (False, 0),  # Оригинальное изображение
        (False, 1),  # Повернуть на 90 градусов
        (False, 2),  # Повернуть на 180 градусов
        (False, 3),  # Повернуть на 270 градусов
        (True, 0),  # Зеркально отразить
        (True, 1),  # Зеркально отразить и повернуть на 90
        (True, 2),  # Зеркально отразить и повернуть на 180
        (True, 3)  # Зеркально отразить и повернуть на 270
)

logging = setting_logs(LOG_FILE)


class YoloDatasetAugmentor:


    def __init__(self, dataset_path, output_path, debug_dir=None, mode="bboxes"):
        """
        Инициализация.

        :param dataset_path: Путь к корневой папке датасета (содержит train, valid, test папки).
        :param output_path: Путь для записи нового аугментированного датасета.
        :param mode: Тип датасета. "bboxes" - обработка боксов, "contours" - обработка контуров.
        """
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.subsets = ["train", "valid", "test"]
        self.mode = mode
        self.class_distributions = defaultdict(Counter)
        self.empty_labels = defaultdict(list)
        self.valid_labels = defaultdict(list)
        if debug_dir:
            self.debug_dir = os.path.join(output_path, debug_dir)
        else:
            self.debug_dir = None

        if mode not in ["bboxes", "contours"]:
            raise ValueError(f"Unsupported mode: {mode}. Use 'bboxes' or 'contours'.")

    def prepare_output_directory(self):
        """
        Создание структуры директорий для нового датасета.
        """
        for subset in self.subsets:
            images_dir = os.path.join(self.output_path, subset, "images")
            labels_dir = os.path.join(self.output_path, subset, "labels")
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)

    def dataset_statistics(self, path):
        """
        Вывод статистики по датасету.
        """
        stats = {}
        total_images = 0

        for subset in self.subsets:
            images_dir = os.path.join(path, subset, "images")
            labels_dir = os.path.join(path, subset, "labels")

            if os.path.exists(images_dir) and os.path.exists(labels_dir):
                image_files = find_all_image_files(images_dir)

                subset_class_distribution = self.class_distributions[subset]
                total_images += len(image_files)

                empty_labels_count = len(self.empty_labels[subset])
                valid_labels_count = len(self.valid_labels[subset])

                stats[f"{subset}_size"] = len(image_files)
                stats[f"{subset}_class_distribution"] = dict(subset_class_distribution)
                stats[f"{subset}_valid_labels"] = valid_labels_count
                stats[f"{subset}_empty_labels"] = empty_labels_count
                stats[f"{subset}_empty_percentage"] = f"{(empty_labels_count / len(image_files) * 100):.2f}" \
                    if image_files else "0.00%"

        # Процентное распределение по split'ам
        for subset in self.subsets:
            stats[f"{subset}_percentage"] = f"{(stats.get(f'{subset}_size', 0) / total_images * 100):.2f}%" \
                if total_images > 0 else "0.00%"

        return stats

    def analyze(self, path):
        """
        Анализирует текущий датасет для сбора статистики.
        """
        self.class_distributions = defaultdict(Counter)
        self.empty_labels = defaultdict(list)
        self.valid_labels = defaultdict(list)
        for subset in self.subsets:
            images_dir = os.path.join(path, subset, "images")
            labels_dir = os.path.join(path, subset, "labels")

            if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
                continue

            image_files = find_all_image_files(images_dir)

            for img_file in image_files:
                img_name = os.path.basename(img_file)
                label_file = os.path.join(labels_dir, f"{Path(img_name).stem}.txt")

                if self.mode == "bboxes":
                    annotations = self.read_yolo_labels(label_file)
                else:
                    annotations = self.read_segmentation_labels(label_file)

                if annotations:
                    self.valid_labels[subset].append(img_file)
                    for annotation in annotations:
                        cls = annotation[0]
                        self.class_distributions[subset][cls] += 1
                else:
                    self.empty_labels[subset].append(img_file)

    def run_augment(self):
        """
        Метод для выполнения аугментаций на всем датасете.
        """
        # Создаем структуру директорий для нового датасета
        self.prepare_output_directory()

        self.analyze(self.dataset_path)
        logging.info("Dataset statistics before augmentation:")
        for key, value in self.dataset_statistics(self.dataset_path).items():
            logging.info(f"{key}: {value}")

        for subset in self.subsets:
            subset_path = os.path.join(self.dataset_path, subset)
            if os.path.exists(subset_path):
                images_dir = os.path.join(subset_path, "images")
                labels_dir = os.path.join(subset_path, "labels")
                output_images_dir = os.path.join(self.output_path, subset, "images")
                output_labels_dir = os.path.join(self.output_path, subset, "labels")
                self.augment_subset(images_dir, labels_dir, output_images_dir, output_labels_dir)

        self.analyze(self.output_path)
        logging.info("Dataset statistics after augmentation:")
        for key, value in self.dataset_statistics(self.output_path).items():
            logging.info(f"{key}: {value}")

    def augment_subset(self, images_dir, labels_dir, output_images_dir, output_labels_dir):
        """
        Аугментация с сохранением результатов в output_path.

        :param images_dir: Папка с оригинальными изображениями.
        :param labels_dir: Папка с оригинальными аннотациями.
        :param output_images_dir: Папка для сохранения обработанных изображений.
        :param output_labels_dir: Папка для сохранения обработанных аннотаций.
        """
        image_paths = find_all_image_files(images_dir)

        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            label_path = os.path.join(labels_dir, f"{Path(img_name).stem}.txt")

            # Чтение изображения и аннотации
            img = cv2.imread(img_path)
            if img is None:
                continue

            if self.mode == "bboxes":
                annotations = self.read_yolo_labels(label_path)
            else:
                annotations = self.read_segmentation_labels(label_path)

            # Аугментация изображений и аннотаций
            for idx, (is_mirrored, n90rotation) in enumerate(TRANSFORMATIONS):
                transformed_img, transformed_annotations = self.apply_transformation(
                    img.copy(), annotations, is_mirrored, n90rotation
                )

                # Генерируем уникальные имена для аугментированных данных
                new_img_name = f"{os.path.splitext(img_name)[0]}_aug_{idx}.jpg"
                new_label_name = f"{Path(new_img_name).stem}.txt"

                # Сохраняем обработанные данные
                cv2.imwrite(os.path.join(output_images_dir, new_img_name), transformed_img)
                if self.mode == "bboxes":
                    self.write_yolo_labels(transformed_annotations, os.path.join(output_labels_dir, new_label_name),
                                           transformed_img)
                else:
                    self.write_segmentation_labels(transformed_annotations,
                                                   os.path.join(output_labels_dir, new_label_name))

                # Отрисовка изображений с аннотациями для теста
                if self.debug_dir:
                    os.makedirs(self.debug_dir, exist_ok=True)
                    debug_img_path = os.path.join(self.debug_dir, new_img_name)
                    if self.mode == "bboxes":
                        # Отрисовываем прямоугольники (bounding boxes)
                        debug_img = self.draw_bboxes(transformed_img, transformed_annotations)
                    else:
                        # Отрисовываем сегментацию
                        debug_img = self.draw_segmentation(transformed_img, transformed_annotations)

                    cv2.imwrite(debug_img_path, debug_img)

    def apply_transformation(self, img, annotations, is_mirrored, n90rotation):
        h, w, _ = img.shape
        if is_mirrored:
            if self.mode == "bboxes":
                annotations = self.flip_annotations_bboxes(annotations)
            else:
                annotations = self.flip_annotations_contours(annotations)

            img = cv2.flip(img, 1)

        for _ in range(n90rotation):
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            if self.mode == "bboxes":
                annotations = self.rotate90clockwise_annotations_bboxes(annotations)
            else:
                annotations = self.rotate90clockwise_annotations_contours(annotations)


        return img, annotations

    def flip_annotations_bboxes(self, bboxes):
        """
        Зеркально отражает боксы относительно вертикальной оси для нормализованных данных.
        """
        flipped = []
        for cls, (x, y, w, h) in bboxes:  # x, y - центр, w, h - размеры
            flipped_x = 1 - x
            flipped.append((cls, (flipped_x, y, w, h)))
        return flipped

    def flip_annotations_contours(self, contours):
        """
        Зеркально отражает контуры относительно вертикальной оси для нормализованных данных.
        """
        flipped = []
        for cls, points in contours:
            flipped_points = [(1 - x, y) for x, y in points]
            flipped.append((cls, flipped_points))
        return flipped

    def rotate90clockwise_annotations_bboxes(self, bboxes):
        """
        Поворачивает боксы на 90 градусов по часовой стрелке для нормализованных координат YOLO.
        """
        rotated = []
        for cls, (x, y, w, h) in bboxes:
            # Конвертируем центр и размеры (YOLO) в координаты углов:
            x1 = x - w / 2  # Левый верхний угол (x1, y1)
            y1 = y - h / 2
            x2 = x + w / 2  # Правый нижний угол (x2, y2)
            y2 = y + h / 2

            # Поворачиваем координаты углов
            new_x1, new_y1 = 1 - y2, x1
            new_x2, new_y2 = 1 - y1, x2

            # Высчитываем новый центр и размеры из повёрнутых углов
            rotated_x = (new_x1 + new_x2) / 2  # Новый x_center
            rotated_y = (new_y1 + new_y2) / 2  # Новый y_center
            rotated_w = abs(new_x2 - new_x1)  # Новый width
            rotated_h = abs(new_y2 - new_y1)  # Новый height

            # Сохраняем повёрнутый бокс
            rotated.append((cls, (rotated_x, rotated_y, rotated_w, rotated_h)))

        return rotated

    def rotate90clockwise_annotations_contours(self, contours):
        """
        Поворачивает контуры на 90 градусов по часовой стрелке для нормализованных данных.
        """
        rotated = []
        for cls, points in contours:
            rotated_points = [(1-y, x) for x, y in points]
            rotated.append((cls, rotated_points))
        return rotated

    def write_yolo_labels(self, bboxes, label_path, img):
        """
        Записывает нормализованные аннотации боксов в YOLO-формате.
        """
        with open(label_path, "w") as f:
            for cls, (x, y, w, h) in bboxes:
                f.write(f"{int(cls)} {x:.10f} {y:.10f} {w:.10f} {h:.10f}\n")

    def write_segmentation_labels(self, contours, label_path):
        """
        Записывает нормализованные контуры в формате: class x1 y1 x2 y2 ... xn yn.
        """
        with open(label_path, "w") as f:
            for cls, points in contours:
                points_str = " ".join(f"{x:.10f} {y:.10f}" for x, y in points)
                f.write(f"{cls} {points_str}\n")

    def read_yolo_labels(self, label_path):
        """
        Чтение YOLO-аннотаций (боксов).
        """
        bboxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    data = line.strip().split(" ")
                    if len(data) == 5:  # YOLO формат: class x_center y_center width height
                        cls = int(data[0])  # Класс объекта
                        bboxes.append((cls, tuple(map(float, data[1:]))))
        return bboxes

    def read_segmentation_labels(self, label_path):
        """
        Чтение аннотаций сегментации с учётом классов.
        """
        contours = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    _line = list(map(float, line.strip().split(" ")))  # Разбиваем строку
                    cls = int(_line[0])  # Первый элемент - это класс
                    points = _line[1:]  # Остальные элементы - координаты
                    if len(points) % 2 == 0:  # Убедимся, что это пары x, y
                        contour = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]
                        contours.append((cls, contour))  # Добавляем класс и контур
        return contours

    def draw_bboxes(self, image, annotations):
        """
        Отрисовка bounding boxes на изображении.

        :param image: Исходное изображение.
        :param annotations: Аннотации YOLO в формате (class_id, (x_center, y_center, width, height), ...)
        :return: Изображение с отрисованными bounding boxes.
        """
        for annotation in annotations:
            class_id = annotation[0]
            x_center, y_center, width, height = annotation[1]
            h, w, _ = image.shape

            # Преобразуем координаты из относительных (YOLO) в абсолютные
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)

            # Проверка границ изображения
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, w - 1), min(y2, h - 1)

            # Случайный цвет для класса
            color = tuple(np.random.randint(0, 255, size=3).tolist())

            # Отрисовка прямоугольника и подписи класса
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"Class {class_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return image

    def draw_segmentation(self, image, annotations):
        """
        Отрисовка сегментации на изображении.

        :param image: Исходное изображение.
        :param annotations: Список координат контуров и ID классов [[class_id, x1, y1, x2, y2, ...], ...].
        :return: Изображение с отрисованной сегментацией.
        """
        h, w, _ = image.shape
        for annotation in annotations:
            class_id = annotation[0]
            points = annotation[1:]

            # Преобразуем нормализованные координаты (YOLO) в пиксели
            contour = [(int(x * w), int(y * h)) for x, y in points[0]]

            # Случайный цвет для каждого класса
            color = tuple(np.random.randint(0, 255, size=3).tolist())

            # Отрисовка контура
            cv2.polylines(image, [np.array(contour, np.int32)], isClosed=True, color=color, thickness=2)
            if contour:
                center_x, center_y = contour[0]
                cv2.putText(image, f"Class {class_id}", (center_x, center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return image

def main():
    parser = argparse.ArgumentParser(description="Скрипт для аугментации датасета YOLO.")
    parser.add_argument('--dataset', type=str, required=True, help='Путь к датасету.')
    parser.add_argument('--output', type=str, required=True, help='Путь для сохранения аугментированного датасета.')
    parser.add_argument('--debug_dir', type=str, help='Папка для сохранения изображений с аннотациями (опционально).')
    parser.add_argument('--mode', type=str, choices=['bboxes', 'contours'], default='bboxes', help='Тип аугментации: "bboxes" или "contours".')

    args = parser.parse_args()
    dataset_path, output_path, debug_dir, mode = args.dataset, args.output, args.debug_dir, args.mode

    logging.info("Running augmentation...")
    logging.info(f"Params: ")
    logging.info(f"dataset - {dataset_path}")
    logging.info(f"output - {output_path}")
    logging.info(f"debug_dir - {debug_dir}")
    logging.info(f"mode - {mode}")


    augmentor = YoloDatasetAugmentor(dataset_path, output_path, debug_dir, mode)
    augmentor.run_augment()

if __name__ == "__main__":
    main()