#!/bin/bash

# Пути к папкам
INPUT_FOLDER="./data/test_data/normal"
TRAIN_IMAGES="./data/train_data"
DARK_IMAGES_FOLDER="./data/test_data/dark"
LIGHT_IMAGES_FOLDER="./data/test_data/light"
OUTPUT_DIR="results"
MODEL_PATH="snapshots/Epoch50.pth"

# Шаг 1: Очистка временных изображений если есть
echo "Очистка временных изображений..."
python cleanup_images.py

if [ $? -ne 0 ]; then
    echo "Ошибка при очистке временных изображений."
    exit 1
fi
# Шаг 2: Подготовка изображений (gamma_correction_processor.py)
echo "Запуск подготовки изображений..."
python gamma_correction_processor.py

if [ $? -ne 0 ]; then
    echo "Ошибка при подготовке изображений. Прерывание выполнения."
    exit 1
fi

# Шаг 3: Запуск тренировки модели (train.py)
echo "Запуск тренировки модели..."
python lowlight_train.py --lowlight_images_path "$TRAIN_IMAGES"

if [ $? -ne 0 ]; then
    echo "Ошибка при тренировке модели. Прерывание выполнения."
    exit 1
fi

# Шаг 4: Тестирование модели и генерация отчёта (test_and_report.py)
echo "Запуск тестирования модели и генерации отчёта..."
python clahe_and_zero.py --input_dir "$INPUT_FOLDER" --output_dir "$OUTPUT_DIR" --model_path "$MODEL_PATH"

if [ $? -ne 0 ]; then
    echo "Ошибка при тестировании модели. Прерывание выполнения."
    exit 1
fi

# Шаг 5: Очистка временных изображений
echo "Очистка временных изображений..."
python cleanup_images.py

if [ $? -ne 0 ]; then
    echo "Ошибка при очистке временных изображений."
    exit 1
fi

echo "Все этапы выполнены успешно!"