import os
import numpy as np
import torch
import torch.utils.data as data
import argparse

from segmentation_loss import Dice_Loss
from segmentation import Unet3D as segnet
from autoencoder import Unet3D_encoder as encoder
from discriminator import discriminator
from ppmi import ppmi_pairs

from time import time
from ultils import *
from train import train_epoch
from val import val_epoch
from ms_ssim import plot_mssim
from vis import vis_image as vis

def main():
	'''Configuration'''
	parser = argparse.ArgumentParser()
	parser.add_argument('--encoder', type=str)
	parser.add_argument('--segmentation', type=str)
	parser.add_argument('--discriminator', type=str)
	parser.add_argument('--batch_size', type=int)
	parser.add_argument('--LAMBDA', type=int)
	parser.add_argument('--save_dir', type=str)
	args = parser.parse_args()
	
	LAMBDA = args.LAMBDA
	batch_size = args.batch_size
	learning_rate = 1e-4
	num_epochs = 500
	
	device = torch.device('cuda')
	
	save_dir = args.save_dir
	if os.path.exists(save_dir) == False:
		os.mkdir(save_dir)

	enc_path = args.encoder
	dis_path = args.discriminator
	seg_path = args.segmentation

	'''Initialize dataset'''
	train_set = ppmi_pairs(mode = 'train')
	val_set = ppmi_pairs(mode = 'val')
	
	'''Initialize networks'''	
	# init model encrypter
	Encrypter = encoder(1,1,16).to(device)
	if os.path.exists(enc_path):
		Encrypter.load_state_dict(torch.load(enc_path, map_location=device))
	Encrypter.train()
	# init discriminator
	Discriminator = discriminator().to(device)
	if os.path.exists(dis_path):
		Discriminator.load_state_dict(torch.load(dis_path, map_location=device))
	Discriminator.train()
	# init segmentator
	Segmentator = segnet(1,6,32).to(device)
	if os.path.exists(seg_path):
		Segmentator.load_state_dict(torch.load(seg_path, map_location=device))
	Segmentator.train()
	#
	models = {'enc': Encrypter, 'seg': Segmentator, 'dis': Discriminator}

	'''Initialize optimizer'''
	# declare loss function
	Segment_criterion = Dice_Loss()
	Discrimination_criterion = torch.nn.CrossEntropyLoss()
	criterions = {'seg': Segment_criterion, 'dis': Discrimination_criterion}

	# init optimizer
	params_es = [{"params": Encrypter.parameters()},{"params": Segmentator.parameters()}]
	optimizer_es = torch.optim.Adam(params_es, lr = learning_rate)

	params_d = [{"params": Discriminator.parameters()}]
	optimizer_d = torch.optim.Adam(params_d, lr = learning_rate)
	
	optimizers = {'es': optimizer_es, 'dis': optimizer_d}
	
	for epoch in range(num_epochs):
		print('|==========================\nEPOCH:{}'.format(epoch + 1))
		
		'''Trainnig'''
		_, _, _, _,\
		models, optimizers = train_epoch(models, optimizers, criterions, LAMBDA, train_set, batch_size,device)
 
		'''Validation'''
		if (epoch + 1) % 1 == 0:
			val_epoch(models, criterions, val_set, batch_size, device)
			
		#save model
		if (epoch + 1) % 1 == 0:
			'''save models'''
			models_dir = os.path.join(save_dir,'models')
			if os.path.exists(models_dir) == False:
				os.mkdir(models_dir)
			torch.save(models['enc'].state_dict(), os.path.join(models_dir, 'enc.pt'))
			torch.save(models['seg'].state_dict(), os.path.join(models_dir, 'seg.pt'))
			torch.save(models['dis'].state_dict(), os.path.join(models_dir, 'dis.pt'))
			torch.save(optimizers['es'].state_dict(), os.path.join(models_dir, 'optim_es.pt'))			
			torch.save(optimizers['dis'].state_dict(), os.path.join(models_dir, 'optim_dis.pt'))			

	return

if __name__ == '__main__':
	main()

