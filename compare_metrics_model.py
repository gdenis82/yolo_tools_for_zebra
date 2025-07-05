#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт для сравнения нескольких обучающих моделей YOLO и создания отчёта в формате Markdown.
Скрипт определяет, какая из обучающих моделей показала лучшие результаты по ключевым показателям:
    - metrics/mAP50-95
    - metrics/mAP50
    - metrics/mAP75
    - metrics/precision
    - metrics/recall
    - metrics/F1-score
    - metrics/F1-confidence

# Example usage:
# python compare_training_model_metrics.py --dir path/to/training/folders --model-type best --dataset path/to/dataset.yaml
"""

import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from utils import setting_logs
import torch
import torchvision
from ultralytics import YOLO, checks

# Set up logging
LOG_FILE = 'compare_model_metrics.log'
logging = setting_logs(LOG_FILE)


def setup_device():
    """
    Set up and return available CUDA devices.

    Returns:
        list: List of available CUDA device indices
    """
    DEVICE = []
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"torchvision version: {torchvision.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        logging.info(f"Number of devices: {num_devices}")

        for i in range(num_devices):
            DEVICE.append(i)
            logging.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        logging.info("No CUDA-compatible devices found. Using CPU.")

    return DEVICE


def find_training_folders(directory):
    """
    Find all training folders in the specified directory.

    Args:
        directory (str): Directory to search for training folders

    Returns:
        list: List of training folder paths
    """
    directory = Path(directory)
    if not directory.exists() or not directory.is_dir():
        logging.error(f"Directory not found or not a directory: {directory}")
        return []

    # Find all folders that have a 'weights' subfolder
    training_folders = []
    for folder in directory.iterdir():
        if folder.is_dir() and (folder / 'weights').exists() and (folder / 'weights').is_dir():
            training_folders.append(folder)

    if not training_folders:
        logging.error(f"No training folders with 'weights' subfolder found in {directory}")
    else:
        logging.info(f"Found {len(training_folders)} training folders in {directory}")

    return training_folders


def find_model_in_folder(folder, model_type):
    """
    Find the specified model type in the weights subfolder.

    Args:
        folder (Path): Training folder path
        model_type (str): Model type ('best' or 'last')

    Returns:
        Path: Path to the model file, or None if not found
    """
    weights_folder = folder / 'weights'
    model_file = weights_folder / f"{model_type}.pt"

    if model_file.exists():
        return model_file
    else:
        logging.warning(f"Model file {model_file} not found in {folder}")
        return None


def calculate_f1_scores(metrics, names, is_box=True):
    """
    Вычисляет баллы F1 на основе показателей и названий классов, вычисляет баллы F1 для каждого класса,
определяя оптимальный порог достоверности для каждого класса, и регистрирует соответствующую
информацию. Функция поддерживает показатели как на основе рамок, так и на основе масок.

    :param metrics: object, содержащий данные о сегментации или метриках на основе блоков для F1
        расчет баллов. Он должен иметь атрибуты "seg" и/или "box" в зависимости от
типа обрабатываемых показателей.
    :param names: dict, в котором ключи — это индексы классов (в виде строк), а значения — названия классов. Они используются для сопоставления данных метрик с конкретными названиями классов.
    :param is_box: bool, указывающий, следует ли рассчитывать F1-оценки на основе блоков  (True) или на основе масок (False). По умолчанию True.

    :return: tuple, содержащий три элемента:
        - Среднее пороговое значение достоверности (с плавающей запятой), рассчитанное для всех классов.
        - Словарь с показателями F1 для каждого класса и соответствующими оптимальными пороговыми значениями достоверности.
        - Общий средний показатель F1 для всех классов.
    """
    # Проверка, не пытаемся ли мы вычислить метрики масок, но в модели их нет
    if not is_box and (not hasattr(metrics, 'seg') or not hasattr(metrics.seg, 'curves_results')):
        logging.warning("Attempted to calculate mask F1 scores but model doesn't have segmentation metrics")
        return 0.0, {}

    metric_type = metrics.box if is_box else metrics.seg

    # Проверка, существует curves_results и имеет ли он ожидаемую структуру
    if not hasattr(metric_type, 'curves_results') or len(metric_type.curves_results) < 2 or len(
            metric_type.curves_results[1]) < 2:
        metric_type_name = "Box" if is_box else "Mask"
        logging.warning(f"No {metric_type_name} curves_results available for F1 calculation")
        return 0.0, {}

    confidences = np.array(metric_type.curves_results[1][0])
    f1_per_class = []
    f1_best_scores_dict = {}
    best_classes_f1_scores = []

    # Получаем наилучшая уверенность и лучший f1_score для каждого класса и вычисляем их среднее значение
    for key, value in names.items():
        # Проверьте, есть ли этот индекс класса в curves_results
        if int(key) >= len(metric_type.curves_results[1][1]):
            logging.warning(f"Class index {key} ({value}) not found in curves_results")
            continue

        f1_scores = np.array(metric_type.curves_results[1][1][int(key)])

        max_f1_index = np.argmax(f1_scores)
        best_f1_score = f1_scores[max_f1_index]
        best_confidence = confidences[max_f1_index]

        f1_per_class.append(best_confidence)
        best_classes_f1_scores.append(best_f1_score)
        f1_best_scores_dict[value] = {'best_score': best_f1_score, 'best_confidence': best_confidence}

        metric_type_name = "Box" if is_box else "Mask"
        logging.info(f"Max {metric_type_name} F1 class:({value}): {best_f1_score}")
        logging.info(f"Optimal {metric_type_name} Confidence Threshold class:({value}): {best_confidence}")

    if not f1_per_class:
        metric_type_name = "Box" if is_box else "Mask"
        logging.warning(f"No valid classes found for {metric_type_name} F1 calculation")
        return 0.0, {}

    mean_f1_confidence = np.mean(f1_per_class)
    metric_type_name = "Box" if is_box else "Mask"
    logging.info(f"All-classes F1 Confidence {metric_type_name} (mean): {mean_f1_confidence}")

    mean_f1_score = np.mean(best_classes_f1_scores)

    return mean_f1_confidence, f1_best_scores_dict, mean_f1_score


def extract_metrics_from_model(train_name, model_path, dataset_path, device, output_dir):
    """
    Извлекает метрики из модели, загрузив ее и выполнив валидацию на наборе данных.

    Аргументы:
        train_name (str): имя модели
        model_path (str): путь к файлу модели
        dataset_path (str): путь к файлу набора данных в формате YAML
        device (list): список индексов устройств для использования
        output_dir (str): каталог для сохранения результатов проверки

    Возвращает:
        dict: словарь, содержащий извлеченные метрики
    """
    try:
        # Load the model
        model = YOLO(model_path)

        # Extract metrics from model checkpoint
        df_metrics = pd.DataFrame(model.ckpt['train_results'])

        # Validate the model
        metrics = model.val(
            data=dataset_path,
            imgsz=640,
            batch=1,
            device=device,
            project=output_dir,
            name=train_name,
            plots=True
        )

        # Extract basic metrics
        extracted_metrics = {
            'model_name': model.model_name,
            'train_dataset': model.ckpt['train_args']['data'],
            'date_train': model.ckpt['date'],
            'box_map': metrics.box.map,
            'box_map50': metrics.box.map50,
            'box_map75': metrics.box.map75,
            'box_precision': metrics.box.mp,
            'box_recall': metrics.box.mr,
        }

        # Calculate F1 scores for boxes
        names = metrics.names
        box_mean_f1_confidence, box_f1_best_scores_dict, box_mean_f1_score = calculate_f1_scores(metrics, names,
                                                                                                 is_box=True)
        extracted_metrics['box_mean_f1_confidence'] = box_mean_f1_confidence
        extracted_metrics['box_f1_best_scores_dict'] = box_f1_best_scores_dict
        extracted_metrics['box_mean_f1_score'] = box_mean_f1_score

        # If segmentation model, add segmentation metrics
        if metrics.task == 'segment' and hasattr(metrics, 'seg'):
            # Check if seg attribute exists and has necessary metrics
            if hasattr(metrics.seg, 'map') and hasattr(metrics.seg, 'map50') and hasattr(metrics.seg, 'map75'):
                extracted_metrics.update({
                    'seg_map': metrics.seg.map,
                    'seg_map50': metrics.seg.map50,
                    'seg_map75': metrics.seg.map75,
                    'seg_precision': metrics.seg.mp,
                    'seg_recall': metrics.seg.mr,
                })

                # Calculate F1 scores for masks
                mask_mean_f1_confidence, mask_f1_best_scores_dict, mask_mean_f1_score = calculate_f1_scores(metrics,
                                                                                                            names,
                                                                                                            is_box=False)
                extracted_metrics['mask_mean_f1_confidence'] = mask_mean_f1_confidence
                extracted_metrics['mask_f1_best_scores_dict'] = mask_f1_best_scores_dict
                extracted_metrics['mask_mean_f1_score'] = mask_mean_f1_score

            else:
                logging.warning(f"Model task is 'segment' but segmentation metrics are not available")
        else:
            logging.info(f"Model task is '{metrics.task}', skipping segmentation metrics")

        # Add training arguments
        extracted_metrics['train_args'] = model.ckpt['train_args']

        # Add speed metrics
        extracted_metrics['inference_time'] = metrics.speed['inference']
        extracted_metrics['fps'] = 1000 / metrics.speed['inference']

        return extracted_metrics, df_metrics

    except Exception as e:
        logging.error(f"Error extracting metrics from {model_path}: {e}")
        return None, None


def get_best_metrics(metrics_dict, df_metrics_dict):
    """
    Парсинг показателей из метрик моделей.

    Аргументы:
        metrics_dict (dict): словарь, в котором имена тренировок сопоставлены с их показателями

    Возвращает:
        dict: словарь, содержащий показатели для каждой тренировки
    """
    best_metrics_dict = {}

    for train_name, metrics in metrics_dict.items():

        train_metrics_epoch = df_metrics_dict[train_name]

        best_metrics = {}

        # Box metrics
        if 'box_map' in metrics:
            best_metrics['metrics/mAP50-95(B)'] = {'value': metrics['box_map'], 'epoch': 'N/A'}
        if 'box_map50' in metrics:
            best_metrics['metrics/mAP50(B)'] = {'value': metrics['box_map50'], 'epoch': 'N/A'}
        if 'box_map75' in metrics:
            best_metrics['metrics/mAP75(B)'] = {'value': metrics['box_map75'], 'epoch': 'N/A'}
        if 'box_precision' in metrics:
            best_metrics['metrics/precision(B)'] = {'value': metrics['box_precision'], 'epoch': 'N/A'}
        if 'box_recall' in metrics:
            best_metrics['metrics/recall(B)'] = {'value': metrics['box_recall'], 'epoch': 'N/A'}
        if 'box_mean_f1_confidence' in metrics:
            best_metrics['metrics/F1-confidence(B)'] = {'value': metrics['box_mean_f1_confidence'], 'epoch': 'N/A'}
        if 'box_mean_f1_score' in metrics:
            best_metrics['metrics/F1-score(B)'] = {'value': metrics['box_mean_f1_score'], 'epoch': 'N/A'}

        # Segmentation metrics
        if 'seg_map' in metrics:
            best_metrics['metrics/mAP50-95(M)'] = {'value': metrics['seg_map'], 'epoch': 'N/A'}
        if 'seg_map50' in metrics:
            best_metrics['metrics/mAP50(M)'] = {'value': metrics['seg_map50'], 'epoch': 'N/A'}
        if 'seg_map75' in metrics:
            best_metrics['metrics/mAP75(M)'] = {'value': metrics['seg_map75'], 'epoch': 'N/A'}
        if 'seg_precision' in metrics:
            best_metrics['metrics/precision(M)'] = {'value': metrics['seg_precision'], 'epoch': 'N/A'}
        if 'seg_recall' in metrics:
            best_metrics['metrics/recall(M)'] = {'value': metrics['seg_recall'], 'epoch': 'N/A'}
        if 'mask_mean_f1_confidence' in metrics:
            best_metrics['metrics/F1-confidence(M)'] = {'value': metrics['mask_mean_f1_confidence'], 'epoch': 'N/A'}
        if 'mask_mean_f1_score' in metrics:
            best_metrics['metrics/F1-score(M)'] = {'value': metrics['mask_mean_f1_score'], 'epoch': 'N/A'}

        # train loss
        if 'train/box_loss' in train_metrics_epoch:
            best_metrics['train/box_loss'] = {'value': train_metrics_epoch['train/box_loss'].to_list()}
        if 'train/seg_loss' in train_metrics_epoch:
            best_metrics['train/seg_loss'] = {'value': train_metrics_epoch['train/seg_loss'].to_list()}
        if 'train/cls_loss' in train_metrics_epoch:
            best_metrics['train/cls_loss'] = {'value': train_metrics_epoch['train/cls_loss'].to_list()}
        if 'train/dfl_loss' in train_metrics_epoch:
            best_metrics['train/dfl_loss'] = {'value': train_metrics_epoch['train/dfl_loss'].to_list()}

        # val loss
        if 'val/box_loss' in train_metrics_epoch:
            best_metrics['val/box_loss'] = {'value': train_metrics_epoch['val/box_loss'].to_list()}
        if 'val/seg_loss' in train_metrics_epoch:
            best_metrics['val/seg_loss'] = {'value': train_metrics_epoch['val/seg_loss'].to_list()}
        if 'val/cls_loss' in train_metrics_epoch:
            best_metrics['val/cls_loss'] = {'value': train_metrics_epoch['val/cls_loss'].to_list()}
        if 'val/dfl_loss' in train_metrics_epoch:
            best_metrics['val/dfl_loss'] = {'value': train_metrics_epoch['val/dfl_loss'].to_list()}

        best_metrics_dict[train_name] = best_metrics

    return best_metrics_dict


def compare_metrics(metrics_dict):
    """
    Сравнение показателей разных тренировок и определение, какая из них показала лучшие результаты.

    Аргументы:
        metrics_dict (дикт): Словарь сопоставляет названия тренировок с их показателями.

    Возвращается:
        dict: Словарь, содержащий результаты сравнения
    """
    comparison = {}

    # Если есть только одна модель, она автоматически становится лучшей
    if len(metrics_dict) == 1:
        train_name = next(iter(metrics_dict))
        train_metrics = metrics_dict[train_name]

        for metric, details in train_metrics.items():
            comparison[metric] = {
                'best_train': [train_name],  # Wrap in list
                'best_value': details['value'],
                'higher_better': True  # Doesn't matter for a single model
            }

        comparison['overall_best'] = [train_name]  # Wrap in list
        return comparison

    # Показатели, при которых чем выше, тем лучше
    higher_better = [
        'metrics/mAP50-95(B)', 'metrics/mAP50(B)', 'metrics/mAP75(B)',
        'metrics/precision(B)', 'metrics/recall(B)', 'metrics/F1-score(B)', 'metrics/F1-confidence(B)',
        'metrics/mAP50-95(M)', 'metrics/mAP50(M)', 'metrics/mAP75(M)',
        'metrics/precision(M)', 'metrics/recall(M)', 'metrics/F1-score(M)', 'metrics/F1-confidence(M)',
    ]

    # Показатели, при которых чем ниже, тем лучше
    lower_better = [
        'val/box_loss', 'val/seg_loss', 'val/cls_loss', 'val/dfl_loss',
        'train/box_loss', 'train/seg_loss', 'train/cls_loss', 'train/dfl_loss'
    ]

    # Получить все доступные показатели по всем тренировочным прогонам
    all_metrics = set()
    for train_metrics in metrics_dict.values():
        all_metrics.update(train_metrics.keys())

    # Сравнить каждый показатель
    for metric in all_metrics:
        comparison[metric] = {}

        # Получить значения этой метрики для всех тренировочных прогонов
        values = {}
        for train_name, train_metrics in metrics_dict.items():
            if metric in train_metrics:
                values[train_name] = train_metrics[metric]['value']

        if not values:
            continue

        # Определить наилучший тренировочный цикл для этого показателя
        best_models = []
        if metric in higher_better:
            best_value = max(values.values())
            best_models = [train for train, val in values.items() if val == best_value]
            comparison[metric]['higher_better'] = True
        elif metric in lower_better:
            best_value = min(values.values())
            best_models = [train for train, val in values.items() if val == best_value]
            comparison[metric]['higher_better'] = False

        comparison[metric]['best_train'] = best_models
        comparison[metric]['best_value'] = best_value

    # Определите наилучший результат обучения на основе показателей
    box_metrics = [
        'metrics/mAP50-95(B)', 'metrics/mAP50(B)', 'metrics/mAP75(B)',
        'metrics/precision(B)', 'metrics/recall(B)', 'metrics/F1-score(B)', 'metrics/F1-confidence(B)'
    ]
    mask_metrics = [
        'metrics/mAP50-95(M)', 'metrics/mAP50(M)', 'metrics/mAP75(M)',
        'metrics/precision(M)', 'metrics/recall(M)', 'metrics/F1-score(M)', 'metrics/F1-confidence(M)'
    ]

    has_mask_metrics = any('metrics/mAP50-95(M)' in metrics_dict[train_name] for train_name in metrics_dict)
    map_metrics = box_metrics + mask_metrics if has_mask_metrics else box_metrics
    available_map_metrics = [m for m in map_metrics if m in comparison]


    if available_map_metrics:
        # Подсчет количества лучших метрик для каждой модели
        best_counts = {train_name: 0 for train_name in metrics_dict.keys()}

        for metric in available_map_metrics:
            if metric in comparison:
                for best_train in comparison[metric]['best_train']:  # Учитываем все равные модели
                    best_counts[best_train] += 1

        # Если нескольким моделям соответствует одинаковое число лучших метрик
        max_count = max(best_counts.values())
        overall_best_candidates = [train for train, count in best_counts.items() if count == max_count]

        # Если кандидатов несколько, выбираем того, кто чаще всего появляется в качестве лидера в метриках
        if len(overall_best_candidates) > 1:
            best_by_priority = {}
            for candidate in overall_best_candidates:
                best_by_priority[candidate] = sum(
                    1 for metric in available_map_metrics
                    if candidate in comparison[metric]['best_train']
                )

            # Определяем модель, которая чаще всего лидировала
            overall_best = max(best_by_priority, key=best_by_priority.get)
        else:
            # Есть только один лидер
            overall_best = overall_best_candidates[0]

        # Сохраняем результаты
        comparison['overall_best'] = overall_best
        comparison['metrics_used'] = available_map_metrics
        comparison['has_mask_metrics'] = has_mask_metrics

    return comparison


def generate_plots(metrics_dict, output_dir, df_metrics_dict=None):
    """
    Создание сравнительных графиков для ключевых показателей.

    Args:
        metrics_dict (dict): словарь, сопоставляющий названия тренировок с их показателями
        output_dir (str): каталог для сохранения графиков
        df_metrics_dict (dict, необязательно): словарь, сопоставляющий названия тренировок с их историей в виде DataFrame
    """
    os.makedirs(output_dir, exist_ok=True)

    # Есть ли у каких-либо моделей показатели сегментации
    has_seg_metrics = any('seg_map' in metrics for metrics in metrics_dict.values())

    # Ключевые показатели для построения графика
    key_metrics = ['box_map', 'box_map50', 'box_map75', 'box_precision', 'box_recall', 'box_mean_f1_confidence',
                   'box_mean_f1_score']

    # Добавить показатели сегментации только в том случае, если они есть хотя бы в одной модели
    if has_seg_metrics:
        key_metrics.extend(
            ['seg_map', 'seg_map50', 'seg_map75', 'seg_precision', 'seg_recall', 'mask_mean_f1_confidence',
             'mask_mean_f1_score'])
        logging.info("Including segmentation metrics in plots")
    else:
        logging.info("No segmentation metrics found, skipping segmentation plots")

    # Сопоставить внутренние названия метрик с отображаемыми именами
    metric_display_names = {
        'box_map': 'Box mAP50-95',
        'box_map50': 'Box mAP50',
        'box_map75': 'Box mAP75',
        'box_precision': 'Box Precision',
        'box_recall': 'Box Recall',
        'box_mean_f1_confidence': 'Box F1 Confidence',
        'box_mean_f1_score': 'Box F1 Score',
        'seg_map': 'Mask mAP50-95',
        'seg_map50': 'Mask mAP50',
        'seg_map75': 'Mask mAP75',
        'seg_precision': 'Mask Precision',
        'seg_recall': 'Mask Recall',
        'mask_mean_f1_confidence': 'Mask F1 Confidence',
        'mask_mean_f1_score': 'Mask F1 Score',
    }

    # При наличии данных построить линейные графики на основе истории обучения
    if df_metrics_dict and len(df_metrics_dict) > 0:

        # Получить доступные метрики в истории обучения
        all_history_metrics = set()
        for df in df_metrics_dict.values():
            all_history_metrics.update(df.columns)

        # Фильтр столбцов epoch и показателей Lr
        all_history_metrics = [m for m in all_history_metrics if m != 'epoch' and not m.startswith('lr/')]

        # Построить линейные графики для каждой метрики
        for metric in all_history_metrics:
            # Пропустить показатели F1 Score, они будут представлены в виде гистограмм
            if 'F1' in metric:
                continue

            plt.figure(figsize=(10, 6))

            for train_name, df in df_metrics_dict.items():
                if metric in df.columns:
                    plt.plot(df['epoch'], df[metric], label=train_name)

            plt.title(f'{metric} vs Epoch')
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True)

            # Сохранить
            plot_path = os.path.join(output_dir, f"{metric.replace('/', '-')}.png")
            plt.savefig(plot_path)
            plt.close()

            logging.info(f"Generated line plot for {metric} at {plot_path}")

    # Обработка данных одной модели для оценки F1 и других показателей, отсутствующих в истории обучения
    if len(metrics_dict) == 1:
        train_name = next(iter(metrics_dict))
        train_metrics = metrics_dict[train_name]

        # Фильтр показателей, которые есть в этой модели
        available_metrics = [m for m in key_metrics if m in train_metrics]

        # Фильтр показателей F1 Score или показателей, которых нет в истории обучения
        if df_metrics_dict and train_name in df_metrics_dict:
            df = df_metrics_dict[train_name]
            available_metrics = [m for m in available_metrics if
                                 'f1' in m.lower() or not any(col.endswith(m.split('_')[-1]) for col in df.columns)]

        if not available_metrics:
            logging.warning("No metrics available for plotting")
            return

        # Создать единый график со всеми показателями для этой модели
        plt.figure(figsize=(10, 8))

        # Сбор значений для доступных показателей
        metric_names = [metric_display_names.get(m, m) for m in available_metrics]
        values = [train_metrics[m] for m in available_metrics]

        # Создание горизонтальной линейчатой диаграммы
        y_pos = np.arange(len(metric_names))
        plt.barh(y_pos, values)
        plt.yticks(y_pos, metric_names)
        plt.xlabel('Value')
        plt.title(f'Metrics for {train_name}')
        plt.tight_layout()

        # Сохранить
        plot_path = os.path.join(output_dir, "model_metrics.png")
        plt.savefig(plot_path)
        plt.close()

        logging.info(f"Generated metrics plot at {plot_path}")

        # Создать отдельные графики для каждой метрики
        for metric in available_metrics:
            plt.figure(figsize=(8, 6))
            plt.bar([train_name], [train_metrics[metric]])
            plt.title(f'{metric_display_names.get(metric, metric)}')
            plt.ylabel(metric_display_names.get(metric, metric))
            plt.tight_layout()

            # Сохранить
            plot_path = os.path.join(output_dir, f"{metric}.png")
            plt.savefig(plot_path)
            plt.close()

            logging.info(f"Generated plot for {metric} at {plot_path}")

        return

    # Обработка нескольких моделей — создание сравнительных гистограмм для показателей F1 Score
    for metric in key_metrics:
        # Пропустить метрики, которые не являются F1 Score или mAP75, если доступна история обучения
        if df_metrics_dict and not ('f1' in metric.lower() or 'map75' in metric.lower()):
            continue

        # Проверить, есть ли этот показатель в обучающем наборе метрик
        if not any(metric in train_metrics for train_metrics in metrics_dict.values()):
            continue

        plt.figure(figsize=(10, 6))

        # Собрать значения этого показателя за все периоды обучения
        train_names = []
        values = []

        for train_name, train_metrics in metrics_dict.items():
            if metric in train_metrics:
                train_names.append(train_name)
                values.append(train_metrics[metric])

        # Создание линейчатого графика
        plt.bar(train_names, values)
        plt.title(f'{metric_display_names.get(metric, metric)} Comparison')
        plt.ylabel(metric_display_names.get(metric, metric))
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Сохранить
        plot_path = os.path.join(output_dir, f"{metric}.png")
        plt.savefig(plot_path)
        plt.close()

        logging.info(f"Generated plot for {metric} at {plot_path}")


def generate_markdown_report(metrics_dict, best_metrics_dict, comparison, output_file, plots_dir):
    """
    Создает и записывает отчет в формате Markdown на основе метрик модели YOLO, лучших метрик и
    данных для сравнения одной или нескольких обученных моделей. Отчет включает подробные метрики модели,
    значения свойств и соответствующие графики, если они доступны.

    :param metrics_dict: словарь, содержащий метрики для одной или нескольких моделей. Ключом является название модели, а значением —
        другой словарь, содержащий различные метрики (например, mAP, точность, полнота) для этой модели.
    :param best_metrics_dict: словарь, содержащий лучшие показатели, достигнутые в разных моделях. Ключ — это название показателя,
 а значение — соответствующее лучшее зафиксированное значение.
    :param comparison: логический флаг, указывающий, следует ли включать в отчет данные о сравнении моделей.
    :param output_file: путь к выходному файлу в формате Markdown, в котором будет сохранен сгенерированный отчет.
    :param plots_dir: путь к каталогу, содержащему графики, связанные с показателями обучения и проверки.
    :return: None
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        # Одна модель
        if len(metrics_dict) == 1:
            train_name = next(iter(metrics_dict))
            metrics = metrics_dict[train_name]

            # Write header
            f.write(f"# YOLO Model Metrics Report: {train_name}\n\n")

            # Write model details
            f.write("## Model Details\n\n")
            f.write("| Property | Value |\n")
            f.write("|----------|-------|\n")
            f.write(f"| Training | {train_name} |\n")
            f.write(f"| Model | {metrics.get('model_name', 'N/A')} |\n")
            f.write(f"| Date | {metrics.get('date_train', 'N/A')} |\n")
            f.write(f"| Training Dataset | {metrics.get('train_dataset', 'N/A')} |\n")
            f.write(f"| Inference Time (ms) | {metrics.get('inference_time', 'N/A')} |\n")
            f.write(f"| FPS | {metrics.get('fps', 'N/A'):.2f} |\n")
            f.write("\n")

            # Write metrics
            f.write("## Model Metrics\n\n")

            # Check if model has segmentation metrics
            has_seg_metrics = any(key.startswith('seg_') or key == 'mask_mean_f1_confidence' for key in metrics.keys())

            # Group metrics by type
            metric_groups = {
                "Box Detection Metrics": [
                    ('box_map', 'Box mAP50-95'),
                    ('box_map50', 'Box mAP50'),
                    ('box_map75', 'Box mAP75'),
                    ('box_precision', 'Box Precision'),
                    ('box_recall', 'Box Recall'),
                    ('box_mean_f1_confidence', 'Box F1 Confidence'),
                    ('box_mean_f1_score', 'Box F1 Score'),
                ]
            }

            # Add segmentation metrics only if they exist
            if has_seg_metrics:
                metric_groups["Mask Segmentation Metrics"] = [
                    ('seg_map', 'Mask mAP50-95'),
                    ('seg_map50', 'Mask mAP50'),
                    ('seg_map75', 'Mask mAP75'),
                    ('seg_precision', 'Mask Precision'),
                    ('seg_recall', 'Mask Recall'),
                    ('mask_mean_f1_confidence', 'Mask F1 Confidence'),
                    ('mask_mean_f1_score', 'Mask F1 Score'),
                ]

            for group_name, group_metrics in metric_groups.items():
                # Check if any metrics in this group exist
                if not any(metric_key in metrics for metric_key, _ in group_metrics):
                    continue

                f.write(f"### {group_name}\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")

                for metric_key, metric_display in group_metrics:
                    if metric_key in metrics:
                        value = metrics[metric_key]
                        f.write(f"| {metric_display} | {str(value)[:8]} |\n")

                f.write("\n")

            # Write F1 Confidence per class
            if 'box_f1_per_class' in metrics:
                f.write("### Box F1 Confidence Per Class\n\n")
                f.write("| Class | F1 Confidence |\n")
                f.write("|-------|----------|\n")

                for class_name, f1_score in metrics['box_f1_per_class'].items():
                    f.write(f"| {class_name} | {f1_score:.5f} |\n")

                f.write("\n")

            if 'mask_f1_per_class' in metrics:
                f.write("### Mask F1 Confidence Per Class\n\n")
                f.write("| Class | F1 Confidence |\n")
                f.write("|-------|----------|\n")

                for class_name, f1_score in metrics['mask_f1_per_class'].items():
                    f.write(f"| {class_name} | {f1_score:.5f} |\n")

                f.write("\n")

            # Write plots section
            f.write("## Metrics Plots\n\n")

            # Add training history plots if available
            history_plot_patterns = [
                'time.png',
                'train-box_loss.png', 'train-cls_loss.png', 'train-dfl_loss.png', 'train-seg_loss.png',
                'val-box_loss.png', 'val-cls_loss.png', 'val-dfl_loss.png', 'val-seg_loss.png',
                'metrics-mAP50(B).png', 'metrics-mAP50(M).png', 'metrics-mAP50-95(B).png', 'metrics-mAP50-95(M).png',
                'metrics-mAP75(B).png', 'metrics-mAP75(M).png',
                'metrics-precision(B).png', 'metrics-precision(M).png', 'metrics-recall(B).png', 'metrics-recall(M).png'
            ]

            # Group history plots by type
            history_plot_groups = {
                "Training Metrics": ['time.png'],
                "Training Loss": ['train-box_loss.png', 'train-cls_loss.png', 'train-dfl_loss.png',
                                  'train-seg_loss.png'],
                "Validation Loss": ['val-box_loss.png', 'val-cls_loss.png', 'val-dfl_loss.png', 'val-seg_loss.png'],
                "Box Metrics": ['metrics-mAP50(B).png', 'metrics-mAP50-95(B).png', 'metrics-mAP75(B).png',
                                'metrics-precision(B).png', 'metrics-recall(B).png', 'box_mean_f1_confidence.png',
                                'box_mean_f1_score.png'],
                "Mask Metrics": ['metrics-mAP50(M).png', 'metrics-mAP50-95(M).png', 'metrics-mAP75(M).png',
                                 'metrics-precision(M).png', 'metrics-recall(M).png', 'mask_mean_f1_confidence.png',
                                 'mask_mean_f1_score.png'],
            }

            for group_name, plot_files in history_plot_groups.items():
                # Check if any plots in this group exist
                existing_plots = [plot_file for plot_file in plot_files
                                  if os.path.exists(os.path.join(plots_dir, plot_file))]

                if not existing_plots:
                    continue

                f.write(f"### {group_name}\n\n")

                for plot_file in existing_plots:
                    plot_path = os.path.join(plots_dir, plot_file)
                    if os.path.exists(plot_path):
                        rel_path = os.path.relpath(plot_path, os.path.dirname(output_file))
                        # Convert Windows backslashes to forward slashes for markdown
                        rel_path = rel_path.replace('\\', '/')
                        # Extract display name from filename
                        display_name = plot_file.replace('.png', '').replace('-', ' ').replace('_', ' ')
                        f.write(f"#### {display_name}\n\n")
                        f.write(f"![{display_name}]({rel_path})\n\n")

            # Check if the combined metrics plot exists
            combined_plot_path = os.path.join(plots_dir, "model_metrics.png")
            if os.path.exists(combined_plot_path):
                rel_path = os.path.relpath(combined_plot_path, os.path.dirname(output_file))
                # Convert Windows backslashes to forward slashes for markdown
                rel_path = rel_path.replace('\\', '/')
                f.write("### All Metrics\n\n")
                f.write(f"![All Metrics]({rel_path})\n\n")

            # Map internal metric names to display names and file names
            metric_mapping = {
                'box_map': ('Box mAP50-95', 'box_map.png'),
                'box_map50': ('Box mAP50', 'box_map50.png'),
                'box_map75': ('Box mAP75', 'box_map75.png'),
                'box_precision': ('Box Precision', 'box_precision.png'),
                'box_recall': ('Box Recall', 'box_recall.png'),
                'box_mean_f1_confidence': ('Box F1 Confidence', 'box_f1_conf.png'),
                'box_mean_f1_score': ('Box F1 Score', 'box_f1_score.png'),
                'seg_map': ('Mask mAP50-95', 'seg_map.png'),
                'seg_map50': ('Mask mAP50', 'seg_map50.png'),
                'seg_map75': ('Mask mAP75', 'seg_map75.png'),
                'seg_precision': ('Mask Precision', 'seg_precision.png'),
                'seg_recall': ('Mask Recall', 'seg_recall.png'),
                'mask_mean_f1_confidence': ('Mask F1 Confidence', 'mask_mean_f1_confidence.png'),
                'mask_mean_f1_score': ('Mask F1 Score', 'mask_mean_f1_score.png'),
            }

            # Group plots by type
            plot_groups = {
                "Box Detection Metrics": ['box_map', 'box_map50', 'box_map75', 'box_precision', 'box_recall',
                                          'box_mean_f1_confidence', 'box_mean_f1_score'],
                "Mask Segmentation Metrics": ['seg_map', 'seg_map50', 'seg_map75', 'seg_precision', 'seg_recall',
                                              'mask_mean_f1_confidence', 'mask_mean_f1_score']
            }

            for group_name, group_metrics in plot_groups.items():
                # Check if any metrics in this group exist
                if not any(metric in metric_mapping and
                           os.path.exists(os.path.join(plots_dir, metric_mapping[metric][1]))
                           for metric in group_metrics):
                    continue

                f.write(f"### {group_name}\n\n")

                for metric in group_metrics:
                    if metric in metric_mapping:
                        display_name, file_name = metric_mapping[metric]
                        plot_path = os.path.join(plots_dir, file_name)

                        if os.path.exists(plot_path):
                            rel_path = os.path.relpath(plot_path, os.path.dirname(output_file))
                            # Convert Windows backslashes to forward slashes for markdown
                            rel_path = rel_path.replace('\\', '/')
                            f.write(f"#### {display_name}\n\n")
                            f.write(f"![{display_name}]({rel_path})\n\n")

            # Write conclusion
            f.write("## Summary\n\n")
            f.write(f"This report presents the metrics for the YOLO model **{train_name}**. ")

            if 'box_mean_f1_confidence' in metrics:
                f.write(f"The model achieved a Box F1 confidence of **{metrics['box_mean_f1_confidence']:.5f}**. ")
            if 'box_mean_f1_score' in metrics:
                f.write(f"The model achieved a Box F1 score of **{metrics['box_mean_f1_score']:.5f}**. ")

            # Only include mask F1 score if it exists
            if 'mask_mean_f1_confidence' in metrics:
                f.write(
                    f"For mask segmentation, it achieved an F1 confidence of **{metrics['mask_mean_f1_confidence']:.5f}**. ")
                if 'mask_mean_f1_score' in metrics:
                    f.write(
                        f"For mask segmentation, it achieved an F1 score of **{metrics['mask_mean_f1_score']:.5f}**. ")
            else:
                f.write(f"This model does not include mask segmentation. ")

            f.write("\n\n")

            # Write recommendations
            f.write("### Recommendations\n\n")
            f.write("Based on these metrics, consider the following recommendations:\n\n")
            f.write("1. Evaluate the model on specific test cases to verify real-world performance\n")
            f.write("2. Compare these metrics with previous versions of the model to track improvements\n")
            f.write("3. Consider fine-tuning the model if certain classes have lower F1 scores\n\n")

        # Множество моделей
        else:
            # Пишем заголовок
            f.write("# Отчет о сравнении моделей обучения YOLO\n\n")

            if 'overall_best' in comparison:
                f.write(f"## Лучшая обученная модель: {comparison['overall_best']}\n\n")

            # Запись деталей тренировок
            f.write("## Детали тренировок\n\n")

            # Создаем таблицу показателей тренировок
            f.write("| Training | Model | Date |\n")
            f.write("|----------|-------|------|\n")

            for train_name, metrics in metrics_dict.items():
                model_name = metrics.get('model_name', 'N/A')
                date_train = metrics.get('date_train', 'N/A')
                f.write(f"| {train_name} | {model_name} | {date_train} |\n")

            f.write("\n")

            f.write("## Сравнение показателей\n\n")

            # Проверяем есть ли данные для сегментации
            has_seg_metrics = any('metrics/mAP50-95(M)' in metrics for metrics in best_metrics_dict.values())

            # Группируем метрики по типу
            metric_groups = {
                "Box Detection Metrics": [
                    ('metrics/mAP50-95(B)', 'Box mAP50-95'),
                    ('metrics/mAP50(B)', 'Box mAP50'),
                    ('metrics/mAP75(B)', 'Box mAP75'),
                    ('metrics/precision(B)', 'Box Precision'),
                    ('metrics/recall(B)', 'Box Recall'),
                    ('metrics/F1-confidence(B)', 'Box F1 Confidence'),
                    ('metrics/F1-score(B)', 'Box F1 Score'),
                ]
            }

            # Добавит показатели сегментации только в том случае, если они есть хотя бы в одной модели
            if has_seg_metrics:
                metric_groups["Mask Segmentation Metrics"] = [
                    ('metrics/mAP50-95(M)', 'Mask mAP50-95'),
                    ('metrics/mAP50(M)', 'Mask mAP50'),
                    ('metrics/mAP75(M)', 'Mask mAP75'),
                    ('metrics/precision(M)', 'Mask Precision'),
                    ('metrics/recall(M)', 'Mask Recall'),
                    ('metrics/F1-confidence(M)', 'Mask F1 Confidence'),
                    ('metrics/F1-score(M)', 'Mask F1 Score'),
                ]
                logging.info("Включение показателей сегментации в отчет")
            else:
                logging.info("Показатели сегментации не найдены, раздел сегментации в отчёте пропущен")

            for group_name, metrics in metric_groups.items():
                # Проверка есть ли в группе показатели
                if not any(metric[0] in comparison for metric in metrics):
                    continue

                f.write(f"### {group_name}\n\n")

                # Create a table for this group of metrics
                f.write("| Metric | " + " | ".join(metrics_dict.keys()) + " | Best Training |\n")
                f.write("|--------|" + "|".join(["------" for _ in metrics_dict]) + "|-------------|\n")

                for metric_key, metric_display in metrics:
                    if metric_key not in comparison:
                        continue

                    f.write(f"| {metric_display} | ")

                    for train_name in metrics_dict.keys():
                        if train_name in best_metrics_dict and metric_key in best_metrics_dict[train_name]:
                            value = best_metrics_dict[train_name][metric_key]['value']

                            # Выделите наилучшее соотношение
                            if comparison[metric_key]['best_train'] == train_name:
                                f.write(f"**{str(value)[:8]}** | ")
                            else:
                                f.write(f"{str(value)[:8]} | ")
                        else:
                            f.write("N/A | ")

                    f.write(f"{comparison[metric_key]['best_train']} |\n")

                f.write("\n")

            # Отображение графиков
            f.write("## Графики \n\n")

            # # Добавить графики истории обучения, если они доступны
            # history_plot_patterns = [
            #     'time.png',
            #     'train-box_loss.png', 'train-cls_loss.png', 'train-dfl_loss.png', 'train-seg_loss.png',
            #     'val-box_loss.png', 'val-cls_loss.png', 'val-dfl_loss.png', 'val-seg_loss.png',
            #     'metrics-mAP50(B).png', 'metrics-mAP50(M).png', 'metrics-mAP50-95(B).png', 'metrics-mAP50-95(M).png',
            #     'metrics-mAP75(B).png', 'metrics-mAP75(M).png',
            #     'metrics-precision(B).png', 'metrics-precision(M).png', 'metrics-recall(B).png', 'metrics-recall(M).png',
            #     'box_mean_f1_confidence.png', 'box_mean_f1_score.png', 'mask_mean_f1_confidence.png', 'mask_mean_f1_score.png'
            # ]

            # Группируем графики по типу
            history_plot_groups = {
                "Training Metrics": ['time.png'],
                "Training Loss": ['train-box_loss.png', 'train-cls_loss.png', 'train-dfl_loss.png',
                                  'train-seg_loss.png'],
                "Validation Loss": ['val-box_loss.png', 'val-cls_loss.png', 'val-dfl_loss.png', 'val-seg_loss.png'],
                "Box Metrics": ['metrics-mAP50(B).png', 'metrics-mAP50-95(B).png', 'metrics-mAP75(B).png',
                                'metrics-precision(B).png', 'metrics-recall(B).png', 'box_mean_f1_confidence.png',
                                'box_mean_f1_score.png'],
                "Mask Metrics": ['metrics-mAP50(M).png', 'metrics-mAP50-95(M).png', 'metrics-mAP75(M).png',
                                 'metrics-precision(M).png', 'metrics-recall(M).png', 'mask_mean_f1_confidence.png',
                                 'mask_mean_f1_score.png'],
            }

            for group_name, plot_files in history_plot_groups.items():
                # Проверьте, есть ли в этой группе графики
                existing_plots = [plot_file for plot_file in plot_files
                                  if os.path.exists(os.path.join(plots_dir, plot_file))]

                if not existing_plots:
                    continue

                f.write(f"### {group_name}\n\n")

                for plot_file in existing_plots:
                    plot_path = os.path.join(plots_dir, plot_file)
                    if os.path.exists(plot_path):
                        rel_path = os.path.relpath(plot_path, os.path.dirname(output_file))

                        rel_path = rel_path.replace('\\', '/')

                        display_name = plot_file.replace('.png', '').replace('-', ' ').replace('_', ' ')
                        f.write(f"#### {display_name}\n\n")
                        f.write(f"![{display_name}]({rel_path})\n\n")

            # Сопоставим внутренние имена метрик с отображаемыми именами и именами файлов
            metric_mapping = {
                'box_map': ('Box mAP50-95', 'box_map.png'),
                'box_map50': ('Box mAP50', 'box_map50.png'),
                'box_map75': ('Box mAP75', 'box_map75.png'),
                'box_precision': ('Box Precision', 'box_precision.png'),
                'box_recall': ('Box Recall', 'box_recall.png'),
                'box_mean_f1_confidence': ('Box F1 Confidence', 'box_mean_f1_confidence.png'),
                'box_mean_f1_score': ('Box F1 Score', 'box_mean_f1_score.png'),
                'seg_map': ('Mask mAP50-95', 'seg_map.png'),
                'seg_map50': ('Mask mAP50', 'seg_map50.png'),
                'seg_map75': ('Mask mAP75', 'seg_map75.png'),
                'seg_precision': ('Mask Precision', 'seg_precision.png'),
                'seg_recall': ('Mask Recall', 'seg_recall.png'),
                'mask_mean_f1_confidence': ('Mask F1 Confidence', 'mask_mean_f1_confidence.png'),
                'mask_mean_f1_score': ('Mask F1 Score', 'mask_mean_f1_score.png'),
            }

            # Группируем графики по типу
            plot_groups = {
                "Box Detection Metrics": ['box_map', 'box_map50', 'box_map75', 'box_precision', 'box_recall',
                                          'mean_f1_confidence', 'mean_f1_score'],
                "Mask Segmentation Metrics": ['seg_map', 'seg_map50', 'seg_map75', 'seg_precision', 'seg_recall',
                                              'mask_f1_confidence', 'mask_f1_score']
            }

            for group_name, metrics in plot_groups.items():
                # Проверка наличия в группе показателей
                if not any(metric in metric_mapping and
                           os.path.exists(os.path.join(plots_dir, metric_mapping[metric][1]))
                           for metric in metrics):
                    continue

                f.write(f"### {group_name}\n\n")

                for metric in metrics:
                    if metric in metric_mapping:
                        display_name, file_name = metric_mapping[metric]
                        plot_path = os.path.join(plots_dir, file_name)

                        if os.path.exists(plot_path):
                            rel_path = os.path.relpath(plot_path, os.path.dirname(output_file))

                            rel_path = rel_path.replace('\\', '/')
                            f.write(f"#### {display_name}\n\n")
                            f.write(f"![{display_name}]({rel_path})\n\n")

            f.write("## Вывод\n\n")

            if 'overall_best' in comparison:
                f.write(f"Анализ показателей модели: **{comparison['overall_best']}** показала наилучшие результаты. ")

                # Какие показатели использовались для сравнения
                if 'metrics_used' in comparison:
                    f.write("Сравнение проводилось на основе следующих показателей: ")
                    f.write(", ".join([m.replace('metrics/', '') for m in comparison['metrics_used']]))
                    f.write(". ")

                # Mention if mask metrics were used or not
                if 'has_mask_metrics' in comparison:
                    if comparison['has_mask_metrics']:
                        f.write(
                            "В этом сравнении учитывались показатели обнаружения рамок и показатели сегментации масок.")
                    else:
                        f.write(
                            "В этом сравнении учитывались только показатели обнаружения рамок, поскольку ни у одной из моделей не было показателей сегментации масок.")

                f.write("\n\n")

                # Ключевые улучшения
                best_train = comparison['overall_best']
                key_improvements = {}

                val_seg_loss_included = False

                # Показатели, по которым этот тренинг был лучшим
                for metric_key, details in comparison.items():
                    if isinstance(details, dict) and 'best_train' in details and best_train in details['best_train']:
                        # Поиск имени метрики
                        display_name = None
                        for group in metric_groups.values():
                            for m_key, m_display in group:
                                if m_key == metric_key:
                                    display_name = m_display
                                    break
                            if display_name:
                                break

                        if not display_name:
                            display_name = metric_key

                        value = details['best_value']

                        if metric_key == 'val/seg_loss':
                            val_seg_loss_included = True

                        # Процент улучшения по сравнению со вторым лучшим результатом
                        second_best_value = None
                        second_best_train = None
                        for train_name, train_metrics in best_metrics_dict.items():
                            if train_name != best_train and metric_key in train_metrics:
                                train_value = train_metrics[metric_key]['value']
                                if (second_best_value is None or
                                        (details['higher_better'] and train_value > second_best_value) or
                                        (not details['higher_better'] and train_value < second_best_value)):
                                    second_best_value = train_value
                                    second_best_train = train_name

                        if second_best_value is not None:
                            if details['higher_better']:
                                improvement = ((value - second_best_value) / second_best_value) * 100
                                if improvement > 0:
                                    key_improvements[metric_key] = (improvement, value, second_best_value,
                                                                    second_best_train)
                            else:
                                improvement = ((second_best_value - value) / second_best_value) * 100
                                if improvement > 0:
                                    key_improvements[metric_key] = (improvement, value, second_best_value,
                                                                    second_best_train)

                # Ключевые улучшения
                f.write("### Основные преимущества:\n\n")
                sorted_improvements = sorted(key_improvements.items(), key=lambda x: x[1][0], reverse=True)

                val_seg_loss_included = False
                val_seg_loss_entry = None

                # Есть ли val/seg_loss в списке ключевых улучшений
                for metric, improvement_data in key_improvements.items():
                    if metric == 'val/seg_loss':
                        val_seg_loss_entry = (metric, improvement_data)
                        val_seg_loss_included = True
                        break

                # Ключевые показатели в результатах
                key_metrics = [
                    'metrics/mAP50-95(B)', 'metrics/mAP50(B)', 'metrics/mAP75(B)',
                    'metrics/precision(B)', 'metrics/recall(B)', 'metrics/F1-score(B)', 'metrics/F1-confidence(B)',
                    'metrics/mAP50-95(M)', 'metrics/mAP50(M)', 'metrics/mAP75(M)',
                    'metrics/precision(M)', 'metrics/recall(M)', 'metrics/F1-score(M)', 'metrics/F1-confidence(M)'
                ]

                # Список отображаемых показателей
                metrics_to_display = []

                # Добавим ключевые показатели, указанные в key_improvements
                for metric in key_metrics:
                    if metric in key_improvements:
                        metrics_to_display.append((metric, key_improvements[metric]))

                for metric, improvement_data in sorted_improvements:
                    if metric not in key_metrics and metric != 'val/seg_loss':
                        metrics_to_display.append((metric, improvement_data))

                # Укажем метрики
                improvements_written = 0
                for metric, (improvement, value, second_value, second_train) in metrics_to_display:
                    # Поиск имени для метрики
                    display_name = None
                    for group in metric_groups.values():
                        for m_key, m_display in group:
                            if m_key == metric:
                                display_name = m_display
                                break
                        if display_name:
                            break

                    if not display_name:
                        display_name = metric

                    if "loss" in metric.lower():
                        f.write(
                            f"- **{display_name}**: {str(value)[:8]} (vs {str(second_value)[:8]} in {second_train}, {improvement:.2f}% ниже)\n")
                    else:
                        f.write(
                            f"- **{display_name}**: {str(value)[:8]} (vs {str(second_value)[:8]} in {second_train}, {improvement:.2f}% выше)\n")

                    improvements_written += 1

                    # Количество отображаемых показателей
                    if improvements_written >= 10:
                        break

                # Добавьте val/seg_loss, если он существует и ещё не был включён
                if val_seg_loss_entry and improvements_written < 10:
                    metric, (improvement, value, second_value, second_train) = val_seg_loss_entry
                    f.write(
                        f"- **val/seg_loss**: {str(value)[:8]} (vs {str(second_value)[:8]} in {second_train}, {improvement:.2f}% ниже)\n")
                # Если val/seg_loss отсутствует в key_improvements, но есть в best_metrics_dict, добавим его
                elif not val_seg_loss_included and 'val/seg_loss' in best_metrics_dict.get(best_train,
                                                                                           {}) and improvements_written < 10:
                    val_seg_loss_value = best_metrics_dict[best_train]['val/seg_loss']['value']
                    f.write(f"- **val/seg_loss**: {str(val_seg_loss_value)[:8]}\n")

                f.write("\n")

    logging.info(f"Generated markdown report at {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Compare YOLO training models and generate a markdown report')
    parser.add_argument('--dir', required=True, help='Directory containing training folders')
    parser.add_argument('--model-type', choices=['best', 'last'], default='best',
                        help='Model type to use (best.pt or last.pt)')
    parser.add_argument('--dataset', required=True, help='Path to dataset YAML file for validation')
    parser.add_argument('--output', help='Output directory for results (default: compare_results in input directory)')

    args = parser.parse_args()

    dir_path = args.dir
    model_type = args.model_type
    output_path = args.output
    dataset_path = args.dataset

    # dir_path = "D:\Projects\ZebraTest\ZEBRA"
    # model_type = "best"
    # output_path = None
    # dataset_path = "D:\Projects\ZebraTest\split_dataset\dataset.yaml"

    checks()

    # Настройка устройства
    device = setup_device()

    # Поиск папок тренировок
    training_folders = find_training_folders(dir_path)
    if not training_folders:
        return

    # Поиск моделей в папках тренировок
    valid_models = []
    for folder in training_folders:
        model_path = find_model_in_folder(folder, model_type)
        if model_path:
            valid_models.append((folder.name, model_path))

    if len(valid_models) < 1:
        logging.error(f"No valid models found in training folders")
        return

    logging.info(f"Found {len(valid_models)} valid models to compare")

    # Настройка папки вывода
    input_dir = Path(dir_path)
    output_dir = Path(output_path) if output_path else input_dir / "compare_results"
    output_dir.mkdir(exist_ok=True, parents=True)

    plots_dir = output_dir / "comparison_plots"
    plots_dir.mkdir(exist_ok=True)

    # Извлекаем метрики для каждой модели
    metrics_dict = {}
    df_metrics_dict = {}

    for train_name, model_path in valid_models:
        logging.info(f"Processing model: {train_name} ({model_path})")

        metrics, df_metrics = extract_metrics_from_model(train_name,
                                                         model_path,
                                                         dataset_path,
                                                         device,
                                                         str(output_dir)
                                                         )

        if metrics:
            metrics_dict[train_name] = metrics
            if df_metrics is not None:
                df_metrics_dict[train_name] = df_metrics

    if not metrics_dict:
        logging.error("No valid metrics extracted from any model")
        return

    # Получаем метрики
    best_metrics_dict = get_best_metrics(metrics_dict, df_metrics_dict)

    # Сравнение метрик
    comparison = compare_metrics(best_metrics_dict)

    # Генерация графиков
    generate_plots(metrics_dict, plots_dir, df_metrics_dict)

    # Генерация отчета в файле markdown
    output_file = output_dir / "model_comparison.md"
    generate_markdown_report(metrics_dict, best_metrics_dict, comparison, output_file, plots_dir)

    logging.info("Comparison completed successfully")


if __name__ == "__main__":
    main()
    # Example usage:
    # python compare_training_model_metrics.py --dir path/to/training/folders --model-type best --dataset path/to/dataset.yaml
