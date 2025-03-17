import torch
import numpy as np
import cv2
from PIL import Image
import torch.utils.data as data
# Функция для получения списка изображений
def populate_train_list(lowlight_images_path):
    import glob
    return glob.glob(lowlight_images_path + "*.png")

class lowlight_loader(data.Dataset):
    def __init__(self, lowlight_images_path):
        self.train_list = populate_train_list(lowlight_images_path)
        self.size = 256
        self.data_list = self.train_list
        print("Total training examples:", len(self.train_list))

    def __getitem__(self, index):
        data_lowlight_path = self.data_list[index]
        # Загружаем низкоосвещённое изображение
        data_lowlight = Image.open(data_lowlight_path).convert('L')
        data_lowlight = data_lowlight.resize((self.size, self.size), Image.LANCZOS)
        data_lowlight_np = np.asarray(data_lowlight, dtype=np.uint8)
        # Применяем CLAHE для создания целевого изображения
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        data_clahe_np = clahe.apply(data_lowlight_np)
        # Преобразуем в тензоры
        data_lowlight = np.asarray(data_lowlight, dtype=np.float32) / 255.0
        data_clahe = np.asarray(data_clahe_np, dtype=np.float32) / 255.0
        data_lowlight = torch.from_numpy(data_lowlight).float().unsqueeze(0)
        data_clahe = torch.from_numpy(data_clahe).float().unsqueeze(0)
        return data_lowlight, data_clahe

    def __len__(self):
        return len(self.data_list)