
"""
Этот скрипт конвертирует аннотации из .json программы X-AnyLabeling в формат YOLO.
"""

import os
import json


def collect_labels(input_dir):
    """Собрать уникальные метки из всех файлов JSON """
    labels = set()
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".json"):
                json_file = os.path.join(root, file)
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for shape in data.get("shapes", []):
                        label = shape.get("label")
                        if label:
                            # Приводим метку к нижнему регистру
                            labels.add(label.lower())
    return sorted(labels)  # Сортируем метки для последовательного назначения индексов


def create_classes_file(labels, output_dir):
    """Создать файл classes.txt из списка меток"""
    classes_file_path = os.path.join(output_dir, "classes.txt")
    with open(classes_file_path, "w", encoding="utf-8") as file:
        for label in labels:
            file.write(f"{label}\n")
    print(f"Список классов сохранён в {classes_file_path}")


def convert_to_yolo_format(shape, image_width, image_height, label_to_id):
    """Конвертировать одну аннотацию в формат YOLO"""
    label = shape["label"].lower()  # Приводим метку к нижнему регистру
    points = shape["points"]

    x_min = min(point[0] for point in points)
    x_max = max(point[0] for point in points)
    y_min = min(point[1] for point in points)
    y_max = max(point[1] for point in points)

    # Центр прямоугольника
    x_center = (x_min + x_max) / 2 / image_width
    y_center = (y_min + y_max) / 2 / image_height

    # Ширина и высота прямоугольника
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height

    # Получаем числовой идентификатор метки
    class_id = label_to_id.get(label, None)
    if class_id is None:
        return None  # Если метка не найдена, пропускаем

    return f"{class_id} {x_center} {y_center} {width} {height}\n"


def process_json_file(json_file, output_dir, label_to_id):
    """Обработка одного JSON файла и генерация соответствующего .txt"""
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    image_width = data["imageWidth"]
    image_height = data["imageHeight"]

    output_lines = []

    for shape in data["shapes"]:
        if shape["shape_type"] == "rectangle":
            yolo_line = convert_to_yolo_format(shape, image_width, image_height, label_to_id)
            if yolo_line:
                output_lines.append(yolo_line)

    # Сохранение .txt файла
    output_file_path = os.path.join(output_dir, os.path.basename(json_file).replace(".json", ".txt"))
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        output_file.writelines(output_lines)


def main(input_dir, output_dir):
    """Основная функция для выполнения всех шагов"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Собрать все уникальные метки
    labels = collect_labels(input_dir)

    # 2. Создать список классов (classes.txt) и построить словарь метка → индекс
    create_classes_file(labels, output_dir)
    label_to_id = {label: idx for idx, label in enumerate(labels)}

    # 3. Пройти по всем JSON и создать YOLO-аннотации
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".json"):
                json_file = os.path.join(root, file)
                process_json_file(json_file, output_dir, label_to_id)
                print(f"Обработано: {json_file}")


if __name__ == "__main__":
    # Укажите директорию с JSON файлами и папку для сохранения .txt файлов
    input_directory = "output_folder"
    output_directory = "export_annotations_data"

    main(input_directory, output_directory)