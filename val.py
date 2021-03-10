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
#from iseg import iseg_pairs

from time import time
from metric import count_predictions, compute_metric, compute_dice_score

def val_epoch(models, criterions, val_set, batch_size, device):
	
	loader = data.DataLoader(val_set, batch_size = batch_size, num_workers = 16, pin_memory=True, shuffle = True)
	
	Encrypter = models['enc'].eval()
	Segmentator = models['seg'].eval()
	Discriminator = models['dis'].eval()
	
	segmentation_criterion = criterions['seg']
	discrimination_criterion = criterions['dis']
	
	run_seg_loss = 0
	run_adv_loss = 0
	dice_score = np.zeros(6)

	TP, FP, TN, FN = 0, 0, 0, 0
	
	start_time = time()
	with torch.no_grad():
		for step, (x, x_ref, y, y_ref,d, d_p, d_n, im, im_ref) in enumerate(loader):
			'''eval here'''
			x, x_ref, y, y_ref,d, d_p = x.to(device), x_ref.to(device), y.to(device), y_ref.to(device),d.to(device), d_p.to(device)
			
			z = Encrypter(x)
			z_ref = Encrypter(x_ref)

			#segmentation
			y_hat = Segmentator(z)
			y_hat_ref = Segmentator(z_ref)
			seg_loss_1 = segmentation_criterion(y_hat, y)
			seg_loss_2 = segmentation_criterion(y_hat_ref, y_ref)
			seg_loss = (seg_loss_1 + seg_loss_2)/2
			run_seg_loss += seg_loss.item()
	
			pred_1 = torch.round(y_hat).detach()
			dice_score += compute_dice_score(pred_1, y)
			pred_2 = torch.round(y_hat_ref).detach()
			dice_score += compute_dice_score(pred_2, y_ref)
			
			#discriminator
			d_z_zref = Discriminator(z, z_ref)
			run_adv_loss += discrimination_criterion(d_z_zref, d).item()
			
			_TP, _FP, _TN, _FN = count_predictions(d_z_zref, d)
			TP += _TP
			FP += _FP
			TN += _TN
			FN += _FN
	
	dur = (time() - start_time)
	seg_loss = run_seg_loss / (step + 1)
	adv_loss = run_adv_loss / (step + 1)
	dis_acc = (TP + TN) / (TP + FP + TN + FN)
	
	dice_score = dice_score / (2 * (step+1))
	
	print('|Validation: ----------------------------')
	print('Seg_loss:{:.4f} | Adv_loss:{:.4f}'.format(seg_loss, -adv_loss))
	print('acc:{:.4f} | TP:{} FP:{} TN:{} FN:{} '.format(dis_acc, TP, FP, TN, FN))
	print('dice_socre:{}'.format(dice_score))
	print('duration:{:.0f}'.format(dur))
	
	return seg_loss, adv_loss\
		   dis_acc, dice_score

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--encoder', type=str)
	parser.add_argument('--segmentation', type=str)
	parser.add_argument('--discriminator', type=str)
	parser.add_argument('--dataset', type=str, default='ppmi')
	args = parser.parse_args()
	
	encoder_path = args.encoder
	discriminator_path = args.discriminator
	segmentation_path = args.segmentation
	
	batch_size = 16
	device = torch.device('cuda')
	if args.dataset == 'ppmi':
		val_set = ppmi_pairs(mode = 'val')
	#if args.dataset == 'iseg':
	#	val_set = iseg_pairs()

	'''Initialize networks'''	
	# init model encrypter
	Encrypter = encoder(1,1,16).to(device)
	Encrypter.load_state_dict(torch.load(encoder_path, map_location=device))
	print('Load:{}'.format(encoder_path))
	
	# init discriminator
	Discriminator = discriminator().to(device)
	Discriminator.load_state_dict(torch.load(discriminator_path, map_location=device))
	print('Load:{}'.format(discriminator_path))

	# init segmentator
	Segmentator = segnet(1,6,32).to(device)
	Segmentator.load_state_dict(torch.load(segmentation_path, map_location=device))
	print('Load:{}'.format(segmentation_path))
	
	#
	models = {'enc': Encrypter, 'seg': Segmentator, 'dis': Discriminator}

	# declare loss function
	Segment_criterion = Dice_Loss()
	Discrimination_criterion = torch.nn.CrossEntropyLoss()
	criterions = {'seg': Segment_criterion, 'dis': Discrimination_criterion}

	seg_loss, adv_loss, dis_acc, dice_score = val_epoch(models, criterions, val_set, batch_size, device)
	
if __name__ == '__main__':
	main()

