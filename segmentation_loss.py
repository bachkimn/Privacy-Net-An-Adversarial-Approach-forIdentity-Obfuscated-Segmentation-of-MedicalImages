import torch
import torch.nn as nn

class Dice_Loss(nn.Module):
	def __init__(self):
		super(Dice_Loss, self).__init__()
	

	def forward(self, pred, target):
		smooth = 1.
		loss = 0

		for i in range(1,pred.shape[1]):
			intersection = (pred[:,i,:,:,:] * target[:,i,:,:,:]).sum()
			union = (pred[:,i,:,:,:] + target[:,i,:,:,:]).sum()
			_loss = 1 - (2 * intersection + smooth)/(union + smooth)
			loss += _loss

		loss = loss / pred.shape[1]

		return loss

