import os
import cv2
import re
import numpy as np

def gamma_correction(image, gamma=0.4):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def process_images(input_dir):
    if not os.path.exists(input_dir):
        print(f"Ошибка: Папка {input_dir} не существует.")
        return

    benign_pattern = re.compile(r"^benign \(\d+\)\.(png|jpg|jpeg)$", re.IGNORECASE)
    for filename in os.listdir(input_dir):
        if benign_pattern.match(filename):
            img_path = os.path.join(input_dir, filename)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Ошибка: Не удалось прочитать изображение {img_path}")
                continue

            for gamma in np.arange(0.5, 1.6, 0.1):
                gamma = round(gamma, 1)
                corrected_image = gamma_correction(image, gamma)
                if corrected_image is None:
                    print(f"Ошибка: Не удалось применить гамма-коррекцию для {filename} с gamma={gamma}")
                    continue

                name, ext = os.path.splitext(filename)
                new_filename = f"{name}_{gamma}{ext}"
                output_path = os.path.join(input_dir, new_filename)
                if not os.path.exists(output_path):
                    if not cv2.imwrite(output_path, corrected_image):
                        print(f"Ошибка: Не удалось сохранить изображение {output_path}")
                else:
                    print(f"Файл уже существует: {output_path}")

def move_images_to_folders(input_folder, dark_images_folder, light_images_folder):
    if not os.path.exists(dark_images_folder):
        os.makedirs(dark_images_folder)
    if not os.path.exists(light_images_folder):
        os.makedirs(light_images_folder)

    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Ошибка: Не удалось прочитать изображение {img_path}")
            continue

        dark_img = gamma_correction(image, 0.5)
        light_img = gamma_correction(image, 1.5)

        dark_output_path = os.path.join(dark_images_folder, filename)
        light_output_path = os.path.join(light_images_folder, filename)

        if not cv2.imwrite(dark_output_path, dark_img):
            print(f"Ошибка: Не удалось сохранить затемненное изображение {dark_output_path}")

        if not cv2.imwrite(light_output_path, light_img):
            print(f"Ошибка: Не удалось сохранить осветленное изображение {light_output_path}")

input_folder = r"./data/train_data/"
test_input_folder = r"./data/test_data/normal"
dark_images_folder = r"./data/test_data/dark"
light_images_folder = r"./data/test_data/light"

process_images(input_folder)
move_images_to_folders(test_input_folder, dark_images_folder, light_images_folder)
