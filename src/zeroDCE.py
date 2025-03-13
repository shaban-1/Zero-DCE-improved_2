import numpy as np
import cv2
import torch
from PIL import Image as PILImage
from model import enhance_net_nopool

class ZeroDCE:
	def __init__(self, model_path, target_brightness=128, device=None):
		self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model = self.load_model(model_path)
		self.target_brightness = target_brightness

	@staticmethod
	def gamma_trans(img, gamma):
		gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
		gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
		return cv2.LUT(img, gamma_table)

	def load_model(self, model_path):
		model = enhance_net_nopool().to(self.device)
		model.load_state_dict(torch.load(model_path, map_location=self.device))
		model.eval()
		return model

	def calculate_gamma(self, img):
		current_brightness = np.mean(img)
		gamma = self.target_brightness / (current_brightness + 1e-6)
		gamma = 1.0
		return gamma

	def enhance_image(self, image_path):
		img = PILImage.open(image_path).convert('L')
		img = img.resize((256, 256), PILImage.LANCZOS)
		img = np.asarray(img, dtype=np.float32) / 255.0
		img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(self.device)

		with torch.no_grad():
			enhanced_image, _, _ = self.model(img)

		enhanced_image = enhanced_image.squeeze().cpu().numpy()

		if enhanced_image.ndim == 3:
			enhanced_image = np.mean(enhanced_image, axis=0)  # RGB -> Grayscale

		enhanced_image = np.clip(enhanced_image, 0, 1)
		enhanced_image_uint8 = (enhanced_image * 255).astype(np.uint8)

		# Вычисляем динамическое значение гаммы
		gamma = self.calculate_gamma(enhanced_image_uint8)
		gamma_corrected = self.gamma_trans(enhanced_image_uint8, gamma)
		gamma_corrected = gamma_corrected.astype(np.float32) / 255.0

		return gamma_corrected
