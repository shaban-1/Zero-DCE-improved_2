import os
import argparse
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.filters import sobel
import torch
import piq
from zeroDCE import ZeroDCE


def calculate_brisque(image):
    if isinstance(image, np.ndarray):
        image = np.clip(image, 0, 1)
    else:
        raise TypeError("Expected image to be a numpy array")

    image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)

    return piq.brisque(image_tensor)


# ЭНТРОПИЯ
def calculate_entropy(image):
    histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    histogram = histogram / np.sum(histogram)
    entropy = -np.sum([p * np.log2(p) for p in histogram if p != 0])
    return entropy

# РЕЗКОСТЬ КРАЁВ
def calculate_edge_intensity(image):
    edges = sobel(image)
    return np.mean(edges)


def calculate_metrics(original, enhanced):
    assert original.shape == enhanced.shape, f"Shapes mismatch: {original.shape} vs {enhanced.shape}"

    mse = np.mean((original - enhanced) ** 2)
    psnr_val = psnr(original, enhanced, data_range=1.0)
    ssim_val = ssim(original, enhanced, data_range=1.0)
    entropy_val = calculate_entropy(enhanced)
    edge_intensity = calculate_edge_intensity(enhanced)
    brisque_val = calculate_brisque(enhanced)

    return mse, psnr_val, ssim_val, entropy_val, edge_intensity, brisque_val


# Обработка изображений
def process_images(input_dir, model_path):
    # Списки для всех метрик
    clahe_mse_list, clahe_psnr_list, clahe_ssim_list = [], [], []
    clahe_entropy_list, clahe_edge_intensity_list, clahe_brisque_list = [], [], []

    zero_dce_mse_list, zero_dce_psnr_list, zero_dce_ssim_list = [], [], []
    zero_dce_entropy_list, zero_dce_edge_intensity_list, zero_dce_brisque_list = [], [], []

    best_combined_ssim = -1
    best_image_data = None

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')) and "_mask" not in filename:
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            original_resized = cv2.resize(image, (256, 256))
            original_float = original_resized.astype(np.float32) / 255.0

            # CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_image = clahe.apply(original_resized)
            clahe_float = clahe_image.astype(np.float32) / 255.0

            # Создание экземпляра ZeroDCE
            zero_dce = ZeroDCE(model_path=model_path)
            zero_dce_image = zero_dce.enhance_image(image_path)

            # Метрики для CLAHE
            clahe_mse, clahe_psnr, clahe_ssim_val, clahe_entropy, clahe_edge_intensity, clahe_brisque = calculate_metrics(
                original_float, clahe_float)
            clahe_mse_list.append(clahe_mse)
            clahe_psnr_list.append(clahe_psnr)
            clahe_ssim_list.append(clahe_ssim_val)
            clahe_entropy_list.append(clahe_entropy)
            clahe_edge_intensity_list.append(clahe_edge_intensity)
            clahe_brisque_list.append(clahe_brisque)


            zero_dce_mse, zero_dce_psnr, zero_dce_ssim_val, zero_dce_entropy, zero_dce_edge_intensity, zero_dce_brisque = calculate_metrics(
                original_float, zero_dce_image)
            zero_dce_mse_list.append(zero_dce_mse)
            zero_dce_psnr_list.append(zero_dce_psnr)
            zero_dce_ssim_list.append(zero_dce_ssim_val)
            zero_dce_entropy_list.append(zero_dce_entropy)
            zero_dce_edge_intensity_list.append(zero_dce_edge_intensity)
            zero_dce_brisque_list.append(zero_dce_brisque)


            delta_ssim = zero_dce_ssim_val - clahe_ssim_val
            delta_psnr = zero_dce_psnr - clahe_psnr
            delta_entropy = zero_dce_entropy - clahe_entropy
            delta_edge_intensity = zero_dce_edge_intensity - clahe_edge_intensity
            delta_brisque = clahe_brisque - zero_dce_brisque  # Чем меньше, тем лучше

            score = (2.0 * delta_ssim) + (1.5 * delta_psnr) + (1.0 * delta_entropy) + (1.0 * delta_edge_intensity) + (
                        1.5 * delta_brisque)

            if score > best_combined_ssim:
                best_combined_ssim = score
                best_image_data = {
                    "filename": filename,
                    "original_image": original_float,
                    "clahe_image": clahe_image,
                    "zero_dce_image": (zero_dce_image * 255).astype(np.uint8),
                    "clahe_metrics": (
                    clahe_mse, clahe_psnr, clahe_ssim_val, clahe_entropy, clahe_edge_intensity, clahe_brisque),
                    "zero_dce_metrics": (
                    zero_dce_mse, zero_dce_psnr, zero_dce_ssim_val, zero_dce_entropy, zero_dce_edge_intensity,
                    zero_dce_brisque)
                }

            print(f"Результаты для {filename}:")
            print(f"  CLAHE -    MSE: {clahe_mse:.4f}, PSNR: {clahe_psnr:.4f}, SSIM: {clahe_ssim_val:.4f}, "
                  f"Entropy: {clahe_entropy:.4f}, Edge Intensity: {clahe_edge_intensity:.4f}, BRISQUE: {clahe_brisque:.4f}")
            print(f"  Zero-DCE - MSE: {zero_dce_mse:.4f}, PSNR: {zero_dce_psnr:.4f}, SSIM: {zero_dce_ssim_val:.4f}, "
                  f"Entropy: {zero_dce_entropy:.4f}, Edge Intensity: {zero_dce_edge_intensity:.4f}, BRISQUE: {zero_dce_brisque:.4f}")
            print("-" * 50)

    return (
    clahe_mse_list, clahe_psnr_list, clahe_ssim_list, clahe_entropy_list, clahe_edge_intensity_list, clahe_brisque_list,
    zero_dce_mse_list, zero_dce_psnr_list, zero_dce_ssim_list, zero_dce_entropy_list, zero_dce_edge_intensity_list,
    zero_dce_brisque_list,
    best_image_data)



def save_best_images(best_image_data, output_dir="best_images"):
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))
    else:
        os.makedirs(output_dir)

    if best_image_data is not None:
        filename = best_image_data["filename"]
        original_image = best_image_data["original_image"]
        clahe_image = best_image_data["clahe_image"]
        zero_dce_image = best_image_data["zero_dce_image"]


        cv2.imwrite(os.path.join(output_dir, "best_clahe_image.png"), clahe_image)
        cv2.imwrite(os.path.join(output_dir, "best_zero_dce_image.png"), zero_dce_image)
        print(f"Сохранено лучшее изображение ({filename}) как best_clahe_image.png и best_zero_dce_image.png")


        clahe_mse, clahe_psnr, clahe_ssim_val, clahe_entropy, clahe_edge_intensity, clahe_brisque = calculate_metrics(
            original_image, clahe_image / 255.0)  # Сравнение с оригиналом
        print(f"  CLAHE -    MSE: {clahe_mse:.4f}, PSNR: {clahe_psnr:.4f}, SSIM: {clahe_ssim_val:.4f}, "
              f"Entropy: {clahe_entropy:.4f}, Edge Intensity: {clahe_edge_intensity:.4f}, BRISQUE: {clahe_brisque:.4f}")

        zero_dce_mse, zero_dce_psnr, zero_dce_ssim_val, zero_dce_entropy, zero_dce_edge_intensity, zero_dce_brisque = calculate_metrics(
            original_image, zero_dce_image / 255.0)
        print(f"  Zero-DCE - MSE: {zero_dce_mse:.4f}, PSNR: {zero_dce_psnr:.4f}, SSIM: {zero_dce_ssim_val:.4f}, "
              f"Entropy: {zero_dce_entropy:.4f}, Edge Intensity: {zero_dce_edge_intensity:.4f}, BRISQUE: {zero_dce_brisque:.4f}")
        print("-" * 50)



def main(input_dir, output_dir, model_path):
    (clahe_mse_list, clahe_psnr_list, clahe_ssim_list, clahe_entropy_list, clahe_edge_intensity_list, clahe_brisque_list,
     zero_dce_mse_list, zero_dce_psnr_list, zero_dce_ssim_list, zero_dce_entropy_list, zero_dce_edge_intensity_list, zero_dce_brisque_list,
     best_image_data) = process_images(input_dir, model_path)

    save_best_images(best_image_data, output_dir)

    if len(clahe_mse_list) > 0:
        clahe_avg_mse = np.mean(clahe_mse_list)
        clahe_avg_psnr = np.mean(clahe_psnr_list)
        clahe_avg_ssim = np.mean(clahe_ssim_list)
        clahe_avg_entropy = np.mean(clahe_entropy_list)
        clahe_avg_edge_intensity = np.mean(clahe_edge_intensity_list)
        clahe_avg_brisque = np.mean(clahe_brisque_list)

        zero_dce_avg_mse = np.mean(zero_dce_mse_list)
        zero_dce_avg_psnr = np.mean(zero_dce_psnr_list)
        zero_dce_avg_ssim = np.mean(zero_dce_ssim_list)
        zero_dce_avg_entropy = np.mean(zero_dce_entropy_list)
        zero_dce_avg_edge_intensity = np.mean(zero_dce_edge_intensity_list)
        zero_dce_avg_brisque = np.mean(zero_dce_brisque_list)

        print("Итоговые результаты (средние значения):")
        print(f"  CLAHE -    MSE: {clahe_avg_mse:.4f}, PSNR: {clahe_avg_psnr:.4f}, SSIM: {clahe_avg_ssim:.4f}, "
              f"Average Entropy: {clahe_avg_entropy:.4f}, Edge Intensity: {clahe_avg_edge_intensity:.4f}, BRISQUE: {clahe_avg_brisque:.4f}")
        print(f"  Zero-DCE - MSE: {zero_dce_avg_mse:.4f}, PSNR: {zero_dce_avg_psnr:.4f}, SSIM: {zero_dce_avg_ssim:.4f}, "
              f"Average Entropy: {zero_dce_avg_entropy:.4f}, Edge Intensity: {zero_dce_avg_edge_intensity:.4f}, BRISQUE: {zero_dce_avg_brisque:.4f}")
    else:
        print("Нет изображений для обработки.")

    results = []
    for image_data in [best_image_data] if isinstance(best_image_data, dict) else best_image_data:
        results.append({
            "filename": image_data["filename"],
            "clahe_metrics": image_data["clahe_metrics"],
            "zero_dce_metrics": image_data["zero_dce_metrics"]
        })
    results.append({
        "filename": "Итоговые результаты (средние значения)",
        "clahe_metrics": (
        clahe_avg_mse, clahe_avg_psnr, clahe_avg_ssim, clahe_avg_entropy, clahe_avg_edge_intensity, clahe_avg_brisque),
        "zero_dce_metrics": (
        zero_dce_avg_mse, zero_dce_avg_psnr, zero_dce_avg_ssim, zero_dce_avg_entropy, zero_dce_avg_edge_intensity,
        zero_dce_avg_brisque)
    })
    generate_html_report(results, results_dir=output_dir, report_filename="report.html")


def generate_html_report(results, results_dir="results", report_filename="report.html"):
    """
    Генерирует HTML-отчёт по результатам обработки.

    Параметры:
      results: список словарей с ключами:
               - "filename": имя исходного файла (например, "malignant (98).png")
               - "clahe_metrics": кортеж метрик для CLAHE (mse, psnr, ssim, entropy, edge_intensity, brisque)
               - "zero_dce_metrics": кортеж метрик для Zero-DCE (mse, psnr, ssim, entropy, edge_intensity, brisque)
      results_dir: папка, где находятся сохранённые изображения
      report_filename: имя выходного HTML-файла отчёта
    """
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Отчёт по обработке изображений</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .section { margin-bottom: 50px; }
        .images { display: flex; gap: 20px; }
        .image-block { text-align: center; }
        .image-block img { max-width: 300px; border: 1px solid #ccc; }
        .metrics { margin-top: 10px; font-family: monospace; font-size: 14px; }
        hr { border: none; border-top: 1px solid #aaa; margin: 40px 0; }
    </style>
</head>
<body>
    <h1>Отчёт по обработке изображений</h1>
"""

    for i, res in enumerate(results):
        if(res == results[-1]):

            m1 = res["clahe_metrics"]

            m = res["zero_dce_metrics"]

            html += ("<div class='metrics'>"
                     f"<strong>CLAHE</strong> - MSE: {m1[0]:.4f}, PSNR: {m1[1]:.4f}, SSIM: {m1[2]:.4f}, "
                     f"Entropy: {m1[3]:.4f}, Edge Intensity: {m1[4]:.4f}, BRISQUE: {m1[5]:.4f}"
                     "</div>\n")

            html += ("<div class='metrics'>"
                     f"<strong>Zero-DCE</strong> - MSE: {m[0]:.4f}, PSNR: {m[1]:.4f}, SSIM: {m[2]:.4f}, "
                     f"Entropy: {m[3]:.4f}, Edge Intensity: {m[4]:.4f}, BRISQUE: {m[5]:.4f}"
                     "</div>\n")

            html += "<hr>\n"
            break

        clahe_file = f"best_clahe_image.png"
        zero_file = f"best_zero_dce_image.png"

        html += f"<div class='section'>\n"
        html += f"<h2>Изображение {res['filename']}</h2>\n"
        html += "<div class='images'>\n"

        original_image_path = os.path.normpath(f"data/test_data/LIME/{res['filename']}")
        image = cv2.imread(original_image_path)
        if image is None:
            print(f"Ошибка: файл {original_image_path} не найден или не может быть загружен.")
        else:
            output_path = os.path.join(results_dir, os.path.basename(original_image_path))
            output_path = os.path.normpath(output_path)
            os.makedirs(results_dir, exist_ok=True)
            cv2.imwrite(output_path, image)
            output_path = os.path.basename(output_path)

        html += "<div class='image-block'>\n"
        html += "<h3>Оригинал</h3>\n"
        html += f"<img src='{output_path}' alt='Оригинал' width='256' height='256'>\n"
        html += "</div>\n"

        # Блок для CLAHE с метриками
        html += "<div class='image-block'>\n"
        html += "<h3>CLAHE</h3>\n"
        html += f"<img src='{clahe_file}' alt='CLAHE'>\n"
        m1 = res["clahe_metrics"]

        html += "</div>\n"

        # Блок для Zero-DCE с метриками
        html += "<div class='image-block'>\n"
        html += "<h3>Zero-DCE</h3>\n"
        html += f"<img src='{zero_file}' alt='Zero-DCE'>\n"
        m = res["zero_dce_metrics"]

        html += "</div>\n"

        html += "</div>\n"
        html += ("<div class='metrics'>"
                 f"<strong>CLAHE</strong> - MSE: {m1[0]:.4f}, PSNR: {m1[1]:.4f}, SSIM: {m1[2]:.4f}, "
                 f"Entropy: {m1[3]:.4f}, Edge Intensity: {m1[4]:.4f}, BRISQUE: {m1[5]:.4f}"
                 "</div>\n")

        html += ("<div class='metrics'>"
                 f"<strong>Zero-DCE</strong> - MSE: {m[0]:.4f}, PSNR: {m[1]:.4f}, SSIM: {m[2]:.4f}, "
                 f"Entropy: {m[3]:.4f}, Edge Intensity: {m[4]:.4f}, BRISQUE: {m[5]:.4f}"
                 "</div>\n")

        html += "<hr>\n"
        html += "</div>\n"

    html += """
</body>
</html>
"""
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, report_filename)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Отчёт сохранён по пути: {report_path}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Enhancement with Zero-DCE and CLAHE + Gamma Correction")
    parser.add_argument("--input_dir", type=str, default="C:/Users/sevda/PycharmProjects/Neural Network/Zero-DCE-improved_1/src/data/test_data/DICM/", help="Путь к директории с изображениями для обработки")
    parser.add_argument("--output_dir", type=str, default="results", help="Путь для сохранения лучших изображений")
    parser.add_argument("--model_path", type=str, default="snapshots/Epoch10.pth", help="Путь к модели Zero-DCE")

    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.model_path)