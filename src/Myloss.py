from torchvision.models.vgg import vgg16
import torch
from torch import nn as nn
from torch.nn import functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomLoss(nn.Module):
	def __init__(self):
		super(CustomLoss, self).__init__()
		self.L_spa = L_spa()
		self.L_exp = L_exp(16, 0.6)
		self.L_TV = L_TV()
		self.L_sa = Sa_Loss()
		self.L_cc = Color_constancy_loss()
		self.L_percept = perception_loss()
		self.L_entropy = nn.KLDivLoss()

	def forward(self, enhanced_image, img_lowlight, A):
		Loss_TV = 100 * self.L_TV(A)
		loss_spa = 2 * torch.mean(self.L_spa(enhanced_image, img_lowlight))
		loss_exp = 1 * torch.mean(self.L_exp(enhanced_image))
		loss_sa =  torch.mean(self.L_sa(enhanced_image))
		loss_cc = 2 * torch.mean(self.L_cc(enhanced_image))
		loss_percept = 2 * torch.mean(self.L_percept(enhanced_image))
		loss = Loss_TV + loss_spa + loss_exp + loss_sa + loss_cc + loss_percept
		return loss


class L_spa(nn.Module):
	def __init__(self):
		super(L_spa, self).__init__()
		kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).to(device).unsqueeze(0).unsqueeze(0)
		kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).to(device).unsqueeze(0).unsqueeze(0)
		kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).to(device).unsqueeze(0).unsqueeze(0)
		kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).to(device).unsqueeze(0).unsqueeze(0)
		self.weight_left = nn.Parameter(kernel_left, requires_grad=False)
		self.weight_right = nn.Parameter(kernel_right, requires_grad=False)
		self.weight_up = nn.Parameter(kernel_up, requires_grad=False)
		self.weight_down = nn.Parameter(kernel_down, requires_grad=False)
		self.pool = nn.AvgPool2d(4)

	def forward(self, org, enhance):
		org = org.to(device)
		enhance = enhance.to(device)

		org_mean = torch.mean(org, 1, keepdim=True)
		enhance_mean = torch.mean(enhance, 1, keepdim=True)

		org_pool = self.pool(org_mean)
		enhance_pool = self.pool(enhance_mean)

		weight_diff = torch.max(torch.FloatTensor([1]).to(device) + 10000 * torch.min(org_pool - torch.FloatTensor([0.3]).to(device),torch.FloatTensor([0]).to(device)),torch.FloatTensor([0.5]).to(device))
		E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).to(device)), enhance_pool - org_pool)

		D_org_left = F.conv2d(org_pool, self.weight_left, padding=1)
		D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
		D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
		D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

		D_enhance_left = F.conv2d(enhance_pool, self.weight_left, padding=1)
		D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
		D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
		D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

		D_left = torch.pow(D_org_left - D_enhance_left, 2)
		D_right = torch.pow(D_org_right - D_enhance_right, 2)
		D_up = torch.pow(D_org_up - D_enhance_up, 2)
		D_down = torch.pow(D_org_down - D_enhance_down, 2)

		E = D_left + D_right + D_up + D_down
		return E

class L_exp(nn.Module):
	def __init__(self, patch_size, mean_val):
		super(L_exp, self).__init__()
		self.pool = nn.AvgPool2d(patch_size)
		self.mean_val = mean_val

	def forward(self, x):
		x = x.to(device)
		mean = self.pool(torch.mean(x, 1, keepdim=True))
		d = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).to(device), 2))
		return d


class Exposure_control_loss(nn.Module):
	def __init__(self, patch_size, mean_val):
		super(Exposure_control_loss, self).__init__()
		self.pool = nn.AvgPool2d(patch_size)
		self.mean_val = mean_val

	def forward(self, x):
		b, c, h, w = x.shape
		x = torch.mean(x, 1, keepdim=True)
		mean = self.pool(x)

		d = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).cuda(), 2))
		return d


class Color_constancy_loss(nn.Module):
	def __init__(self):
		super(Color_constancy_loss, self).__init__()

	def forward(self, x):
		b, c, h, w = x.shape
		mean_rgb = torch.mean(x, [2, 3], keepdim=True)
		mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
		Drg = torch.pow(mr - mg, 2)
		Drb = torch.pow(mr - mb, 2)
		Dgb = torch.pow(mb - mg, 2)
		k = torch.mean(torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5))
		return k

class L_TV(nn.Module):
	def __init__(self,TVLoss_weight=1):
		super(L_TV,self).__init__()
		self.TVLoss_weight = TVLoss_weight

	def forward(self,x):
		batch_size = x.size()[0]
		h_x = x.size()[2]
		w_x = x.size()[3]
		count_h =  (x.size()[2]-1) * x.size()[3]
		count_w = x.size()[2] * (x.size()[3] - 1)
		h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
		w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
		return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

class Sa_Loss(nn.Module):
	def __init__(self):
		super(Sa_Loss, self).__init__()

	def forward(self, x ):
		b,c,h,w = x.shape
		r,g,b = torch.split(x , 1, dim=1)
		mean_rgb = torch.mean(x,[2,3],keepdim=True)
		mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
		Dr = r-mr
		Dg = g-mg
		Db = b-mb
		k =torch.pow( torch.pow(Dr,2) + torch.pow(Db,2) + torch.pow(Dg,2),0.5)
		k = torch.mean(k)
		return k

class perception_loss(nn.Module):
	def __init__(self):
		super(perception_loss, self).__init__()
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		features = vgg16(pretrained=True).features.to(device)

		self.to_relu_1_2 = nn.Sequential()
		self.to_relu_2_2 = nn.Sequential()
		self.to_relu_3_3 = nn.Sequential()
		self.to_relu_4_3 = nn.Sequential()

		for x in range(4):
			self.to_relu_1_2.add_module(str(x), features[x])
		for x in range(4, 9):
			self.to_relu_2_2.add_module(str(x), features[x])
		for x in range(9, 16):
			self.to_relu_3_3.add_module(str(x), features[x])
		for x in range(16, 23):
			self.to_relu_4_3.add_module(str(x), features[x])

		# don't need the gradients, just want the features
		for param in self.parameters():
			param.requires_grad = False

		self.device = device

	def forward(self, x):
		x = x.to(self.device)
		h = self.to_relu_1_2(x)
		h = self.to_relu_2_2(h)
		h = self.to_relu_3_3(h)
		h = self.to_relu_4_3(h)
		return h
