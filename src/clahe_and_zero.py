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


# Вычисление BRISQUE
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


# Вычисление метрик
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
def process_images(input_dir, model, max_images=10):
    # Списки для хранения метрик
    clahe_mse_list, clahe_psnr_list, clahe_ssim_list = [], [], []
    clahe_entropy_list, clahe_edge_intensity_list, clahe_brisque_list = [], [], []
    zero_dce_mse_list, zero_dce_psnr_list, zero_dce_ssim_list = [], [], []
    zero_dce_entropy_list, zero_dce_edge_intensity_list, zero_dce_brisque_list = [], [], []

    # Список для хранения данных всех изображений
    image_data_list = []

    image_count = 0
    for filename in os.listdir(input_dir):
        if image_count >= max_images:
            break
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')) and "_mask" not in filename:
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            original_resized = cv2.resize(image, (256, 256))
            original_float = original_resized.astype(np.float32) / 255.0

            # Применение CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_image = clahe.apply(original_resized)
            clahe_float = clahe_image.astype(np.float32) / 255.0

            zero_dce = ZeroDCE(model_path=args.model_path)
            zero_dce_image = zero_dce.enhance_image(image_path)

            # Метрики для CLAHE
            clahe_mse, clahe_psnr, clahe_ssim, clahe_entropy, clahe_edge, clahe_brisque = calculate_metrics(
                original_float, clahe_float)
            clahe_mse_list.append(clahe_mse)
            clahe_psnr_list.append(clahe_psnr)
            clahe_ssim_list.append(clahe_ssim)
            clahe_entropy_list.append(clahe_entropy)
            clahe_edge_intensity_list.append(clahe_edge)
            clahe_brisque_list.append(clahe_brisque)

            # Метрики для Zero-DCE
            zero_dce_mse, zero_dce_psnr, zero_dce_ssim, zero_dce_entropy, zero_dce_edge, zero_dce_brisque = calculate_metrics(
                original_float, zero_dce_image)
            zero_dce_mse_list.append(zero_dce_mse)
            zero_dce_psnr_list.append(zero_dce_psnr)
            zero_dce_ssim_list.append(zero_dce_ssim)
            zero_dce_entropy_list.append(zero_dce_entropy)
            zero_dce_edge_intensity_list.append(zero_dce_edge)
            zero_dce_brisque_list.append(zero_dce_brisque)

            # Сохранение данных изображения вместе с метриками
            image_data_list.append({
                "filename": filename,
                "original_image": original_float,
                "clahe_image": clahe_image,
                "zero_dce_image": zero_dce_image,
                "clahe_metrics": (clahe_mse, clahe_psnr, clahe_ssim, clahe_entropy, clahe_edge, clahe_brisque),
                "zero_dce_metrics": (
                zero_dce_mse, zero_dce_psnr, zero_dce_ssim, zero_dce_entropy, zero_dce_edge, zero_dce_brisque)
            })

            print(f"Результаты для {filename}:")
            print(f"  CLAHE -    MSE: {clahe_mse:.4f}, PSNR: {clahe_psnr:.4f}, SSIM: {clahe_ssim:.4f}, "
                  f"Entropy: {clahe_entropy:.4f}, Edge Intensity: {clahe_edge:.4f}, BRISQUE: {clahe_brisque:.4f}")
            print(f"  Zero-DCE - MSE: {zero_dce_mse:.4f}, PSNR: {zero_dce_psnr:.4f}, SSIM: {zero_dce_ssim:.4f}, "
                  f"Entropy: {zero_dce_entropy:.4f}, Edge Intensity: {zero_dce_edge:.4f}, BRISQUE: {zero_dce_brisque:.4f}")
            print("-" * 50)
            image_count += 1

    return (
    clahe_mse_list, clahe_psnr_list, clahe_ssim_list, clahe_entropy_list, clahe_edge_intensity_list, clahe_brisque_list,
    zero_dce_mse_list, zero_dce_psnr_list, zero_dce_ssim_list, zero_dce_entropy_list, zero_dce_edge_intensity_list,
    zero_dce_brisque_list, image_data_list)


# Сохранение всех изображений в папку results с выводом метрик
def save_all_images(image_data_list, output_dir="results"):
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))
    else:
        os.makedirs(output_dir)

    for idx, image_data in enumerate(image_data_list):
        filename = image_data["filename"]
        original_image = (image_data["original_image"] * 255).astype(np.uint8)
        clahe_image = image_data["clahe_image"]
        zero_dce_image = (image_data["zero_dce_image"] * 255).astype(np.uint8)

        clahe_mse, clahe_psnr, clahe_ssim, clahe_entropy, clahe_edge, clahe_brisque = image_data["clahe_metrics"]
        zero_dce_mse, zero_dce_psnr, zero_dce_ssim, zero_dce_entropy, zero_dce_edge, zero_dce_brisque = image_data[
            "zero_dce_metrics"]

        original_filename = f"original_{idx}_{filename}"
        clahe_filename = f"clahe_{idx}_{filename}"
        zero_dce_filename = f"zero_dce_{idx}_{filename}"
        cv2.imwrite(os.path.join(output_dir, original_filename), original_image)
        cv2.imwrite(os.path.join(output_dir, clahe_filename), clahe_image)
        cv2.imwrite(os.path.join(output_dir, zero_dce_filename), zero_dce_image)

        # Save histograms
        save_histogram(original_image, os.path.join(output_dir, f"hist_original_{idx}_{filename}.png"))
        save_histogram(clahe_image, os.path.join(output_dir, f"hist_clahe_{idx}_{filename}.png"))
        save_histogram(zero_dce_image, os.path.join(output_dir, f"hist_zero_dce_{idx}_{filename}.png"))

        print(f"Сохранены изображения для {filename} в папке {output_dir}:")
        print(f"  {original_filename}")
        print(f"  {clahe_filename} -    MSE: {clahe_mse:.4f}, PSNR: {clahe_psnr:.4f}, SSIM: {clahe_ssim:.4f}, "
              f"Entropy: {clahe_entropy:.4f}, Edge Intensity: {clahe_edge:.4f}, BRISQUE: {clahe_brisque:.4f}")
        print(f"  {zero_dce_filename} - MSE: {zero_dce_mse:.4f}, PSNR: {zero_dce_psnr:.4f}, SSIM: {zero_dce_ssim:.4f}, "
              f"Entropy: {zero_dce_entropy:.4f}, Edge Intensity: {zero_dce_edge:.4f}, BRISQUE: {zero_dce_brisque:.4f}")
        print("-" * 50)



# Основная функция
def main(input_dir, output_dir, model_path):
    zero_dce = ZeroDCE(model_path)

    (
    clahe_mse_list, clahe_psnr_list, clahe_ssim_list, clahe_entropy_list, clahe_edge_intensity_list, clahe_brisque_list,
    zero_dce_mse_list, zero_dce_psnr_list, zero_dce_ssim_list, zero_dce_entropy_list, zero_dce_edge_intensity_list,
    zero_dce_brisque_list, image_data_list) = process_images(input_dir, zero_dce, max_images=10)

    save_all_images(image_data_list, output_dir)

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
        print(
            f"  Zero-DCE - MSE: {zero_dce_avg_mse:.4f}, PSNR: {zero_dce_avg_psnr:.4f}, SSIM: {zero_dce_avg_ssim:.4f}, "
            f"Average Entropy: {zero_dce_avg_entropy:.4f}, Edge Intensity: {zero_dce_avg_edge_intensity:.4f}, BRISQUE: {zero_dce_avg_brisque:.4f}")
    else:
        print("Нет изображений для обработки.")

    results = []
    for image_data in image_data_list:
        results.append({
            "filename": image_data["filename"],
            "clahe_metrics": image_data["clahe_metrics"],
            "zero_dce_metrics": image_data["zero_dce_metrics"]
        })
    results.append({
        "filename": "Итоговые результаты (средние значения)",
        "clahe_metrics": (clahe_avg_mse, clahe_avg_psnr, clahe_avg_ssim, clahe_avg_entropy, clahe_avg_edge_intensity, clahe_avg_brisque),
        "zero_dce_metrics": (zero_dce_avg_mse, zero_dce_avg_psnr, zero_dce_avg_ssim, zero_dce_avg_entropy, zero_dce_avg_edge_intensity, zero_dce_avg_brisque)
    })
    generate_html_report(results, results_dir=output_dir, report_filename="report.html")

import matplotlib.pyplot as plt

def save_histogram(image, filename):
    plt.figure(figsize=(6, 4))
    plt.hist(image.flatten(), bins=256, range=(0, 256), color='blue', alpha=0.7)
    plt.title('Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.savefig(filename)
    plt.close()


def generate_html_report(results, results_dir="results", report_filename="report.html"):
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
        if res == results[-1]:
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

        orig_file = f"original_{i}_{res['filename']}"
        clahe_file = f"clahe_{i}_{res['filename']}"
        zero_file = f"zero_dce_{i}_{res['filename']}"
        hist_orig_file = f"hist_original_{i}_{res['filename']}.png"
        hist_clahe_file = f"hist_clahe_{i}_{res['filename']}.png"
        hist_zero_file = f"hist_zero_dce_{i}_{res['filename']}.png"

        html += f"<div class='section'>\n"
        html += f"<h2>Изображение {res['filename']}</h2>\n"
        html += "<div class='images'>\n"

        # Блок для оригинала
        html += "<div class='image-block'>\n"
        html += "<h3>Оригинал</h3>\n"
        html += f"<img src='{orig_file}' alt='Оригинал'>\n"
        html += f"<img src='{hist_orig_file}' alt='Histogram'>\n"
        html += "</div>\n"

        # Блок для Zero-DCE с метриками
        html += "<div class='image-block'>\n"
        html += "<h3>Zero-DCE</h3>\n"
        html += f"<img src='{zero_file}' alt='Zero-DCE'>\n"
        html += f"<img src='{hist_zero_file}' alt='Histogram'>\n"
        m = res["zero_dce_metrics"]
        html += "</div>\n"

        # Блок для CLAHE с метриками
        html += "<div class='image-block'>\n"
        html += "<h3>CLAHE</h3>\n"
        html += f"<img src='{clahe_file}' alt='CLAHE'>\n"
        html += f"<img src='{hist_clahe_file}' alt='Histogram'>\n"
        m1 = res["clahe_metrics"]
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
    parser.add_argument("--input_dir", type=str,
                        default="./data/test_data/light", #normal_maligant_benign_data
                        help="Путь к директории с изображениями")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Путь для сохранения обработанных изображений")
    parser.add_argument("--model_path", type=str, default="snapshots/Epoch49.pth", help="Путь к модели Zero-DCE")

    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.model_path)