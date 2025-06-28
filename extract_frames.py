
"""
Описание: Этот скрипт предназначен для извлечения кадров из видеофайлов.

Использование: python extract_frames.py --video path_to_video --output output_folder
param:
 - video: Путь к видео файлам или архиву (zip)
 - output: Папка для сохранения кадров.
"""

import os
import zipfile
import argparse
import subprocess

from tqdm import tqdm
from pathlib import Path
from utils import setting_logs, find_all_video_files, extract_zip_file

DEFAULT_OUTPUT_FOLDER = 'frames'

LOG_FILE = 'frames_extraction.log'
logging = setting_logs(LOG_FILE)


def extract_frames_ffmpeg(video_path, output_folder, frames_per_second=1):
    """
    Извлечет кадры из видео файла с помощью FFmpeg и сохраните в указанной папке.


    :param video_path (str): Путь к входному видеофайлу.
    :param output_folder (str): Путь к папке, в которой будут храниться извлеченные кадры.
    :param frames_per_second (int, optional): Количество извлекаемых кадров в секунду видео.
    """
    video_name = Path(video_path).name
    frame_filename_pattern = os.path.join(output_folder, f"{video_name}_frame_%04d.jpg")

    # Строим команду FFmpeg для извлечения кадров
    command = [
        'ffmpeg', '-i', video_path, '-vf', f'fps={frames_per_second}', frame_filename_pattern
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Ошибка при обработке видео с помощью FFmpeg: {e}")

def process_directory(directory, output_folder):
    """
    Обрабатывает все видеофайлы в указанной директории.

    :param directory: Директория для поиска видеофайлов.
    :param output_folder: Папка для сохранения кадров.
    """

    video_files = find_all_video_files(directory)
    if not video_files:
        logging.warning(f"В директории {directory} не найдено поддерживаемых видео файлов.")
        return

    os.makedirs(output_folder, exist_ok=True)

    for video_path in tqdm(video_files, total=len(video_files), desc="Обрабатываем видео"):
            extract_frames_ffmpeg(video_path, output_folder)

    logging.info("Обработка видео завершена.")


def main():
    """
    Основная функция для обработки аргументов и запуска извлечения кадров.
    """
    parser = argparse.ArgumentParser(description='Извлечение кадров из видео файлов.')
    parser.add_argument('--video', type=str, help='Путь к видео файлам или архиву (zip).')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_FOLDER, help='Папка для сохранения кадров.')

    args = parser.parse_args()

    video_path = args.video
    output_folder = args.output

    logging.info("Running frames extraction script...")
    logging.info(f"Params: ")
    logging.info(f"video={video_path}")
    logging.info(f"output={output_folder}")

    if not video_path:
        error_message = "Ошибка: Директория к видео не указана."
        logging.error(error_message)
        raise FileNotFoundError(error_message)

    # Если это архив
    if zipfile.is_zipfile(video_path):
        extract_zip_file(video_path, logging)
        # После распаковки, ищем видеофайлы в распакованной папке
        extracted_folder = Path(video_path).parent
        process_directory(extracted_folder, output_folder)

    # Если это директория с видео файлами
    elif os.path.isdir(video_path):
        process_directory(video_path, output_folder)

    # Если это одиночный видеофайл
    elif os.path.isfile(video_path):
        extract_frames_ffmpeg(video_path, output_folder)

    else:
        error_message = f"Ошибка: {video_path} не является допустимым файлом."
        logging.error(error_message)
        raise FileNotFoundError(error_message)


if __name__ == "__main__":
    main()
