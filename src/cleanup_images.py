import os
import re
import shutil

def cleanup_generated_images(dark_output_dir, light_output_dir, train_output_data):
    benign_pattern = re.compile(r"^benign \(\d+\)\.(png|jpg|jpeg)$", re.IGNORECASE)
    for folder in [dark_output_dir, light_output_dir, train_output_data]:
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                if not benign_pattern.match(filename):
                    file_path = os.path.join(folder, filename)
                    try:
                        os.remove(file_path)
                        print(f"Удален файл: {file_path}")
                    except Exception as e:
                        print(f"Ошибка при удалении {file_path}: {e}")
        else:
            print(f"Папка {folder} не существует, пропуск.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Очистка временных изображений и папки results")
    parser.add_argument("--dark_output_dir", type=str, default="./data/test_data/dark", help="Путь к папке с затемненными изображениями")
    parser.add_argument("--light_output_dir", type=str, default="./data/test_data/light", help="Путь к папке с осветленными изображениями")
    parser.add_argument("--input_folder", type=str, default="./data/train_data/", help="Путь к папке results")

    args = parser.parse_args()
    cleanup_generated_images(args.dark_output_dir, args.light_output_dir, args.input_folder)