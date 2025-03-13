import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import glob
import random

random.seed(1143)

def populate_train_list(lowlight_images_path):
	image_list_lowlight = glob.glob(
		os.path.join(lowlight_images_path, "*.png"),
		recursive=True
	)
	image_list_lowlight = [f for f in image_list_lowlight if "_mask" not in f]
	random.shuffle(image_list_lowlight)
	return image_list_lowlight

class lowlight_loader(data.Dataset):

	def __init__(self, lowlight_images_path):
		self.train_list = populate_train_list(lowlight_images_path)
		self.size = 256

		self.data_list = self.train_list
		print("Total training examples:", len(self.train_list))

	def __getitem__(self, index):
		data_lowlight_path = self.data_list[index]
		data_lowlight = Image.open(data_lowlight_path).convert('L')
		data_lowlight = data_lowlight.resize((self.size, self.size), Image.LANCZOS)
		data_lowlight = np.asarray(data_lowlight, dtype=np.float32) / 255.0
		data_lowlight = torch.from_numpy(data_lowlight).float()
		return data_lowlight.unsqueeze(0)  # Получаем тензор размерности (1, H, W)

	def __len__(self):
		return len(self.data_list)