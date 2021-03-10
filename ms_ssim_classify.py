import argparse
import os
import torch
import numpy as np
from time import time
from skimage.measure import compare_ssim as ssim
from autoencoder import Unet3D_encoder as encoder
import torch.utils.data as data
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 28})
from ppmi import ppmi_pairs

def plot_dist(dist1,dist2):
	dist1 = np.array(dist1)
	dist2 = np.array(dist2)
	s = min(dist1.min(),dist2.min())
	e = max(dist1.max(),dist2.max())
	bins = np.linspace(s,e,100)
	plt.hist(dist1,bins=bins, alpha = 0.5, density=True, color = 'r', label='intra-subject')
	plt.hist(dist2,bins=bins, alpha = 0.5, density=True, color = 'b', label='inter-subject')
	plt.legend(loc='upper left')
	plt.show()
	return

def find_ms_ssim_threshold(pos_ms_ssim, neg_ms_ssim):
	pos_mean = sum(pos_ms_ssim) / len(pos_ms_ssim)
	neg_mean = sum(neg_ms_ssim) / len(neg_ms_ssim)
	step = (pos_mean - neg_mean) / 100
	acc_max = 0.5
	thres = neg_mean

	for i in range(1, 101):
		_thres = neg_mean + step * i 
		acc, TP, FP, TN, FN = classify_ms_ssim(pos_ms_ssim, neg_ms_ssim, _thres)
		if acc > acc_max:
			acc_max = acc
			thres = _thres

	return thres, acc_max

def compute_ms_ssim(dataset, encryptor = None, device = None):
	if encryptor != None:
		Encryptor = encoder(1,1,16).to(device)
		Encryptor.load_state_dict(torch.load(encryptor, map_location=device))
		Encryptor.eval()
	
	loader = data.DataLoader(dataset, batch_size = 1, num_workers = 16, pin_memory=True, shuffle = True)
	
	pos_ms_ssim = []
	neg_ms_ssim = []
	start = time()
	for step, (x, x_ref, y, y_ref,d, d_p, d_n, im, im_ref) in enumerate(loader):
		if encryptor != None:
			with torch.no_grad():
				x, x_ref = x.to(device), x_ref.to(device)
				x = Encryptor(x).cpu()
				x_ref = Encryptor(x_ref).cpu()
			
		x = x[0,0,:,:,:].numpy().astype(float)
		x_ref = x_ref[0,0,:,:,:].numpy().astype(float)
		ms_ssim_score = ssim(x, x_ref)
		if d.item() == 1:
			pos_ms_ssim += [ms_ssim_score]
		elif d.item() == 0:
			neg_ms_ssim += [ms_ssim_score]
	dur = time() - start
	print('|- Compute ms-ssim scores, duration: {:.0f} sec'.format(dur))
	return pos_ms_ssim, neg_ms_ssim
	
def classify_ms_ssim(pos_ms_ssim, neg_ms_ssim, thres):
	pos_ms_ssim = np.array(pos_ms_ssim)
	neg_ms_ssim = np.array(neg_ms_ssim)
	
	TP = len(np.where(pos_ms_ssim > thres)[0])
	FP = len(np.where(pos_ms_ssim <= thres)[0])
	
	TN = len(np.where(neg_ms_ssim <= thres)[0])
	FN = len(np.where(neg_ms_ssim > thres)[0])
	
	acc = (TP + TN) / (TP + FP + TN + FN) 
	
	return acc, TP, FP, TN, FN
	
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--encoder', type = str, default = None)
	args = parser.parse_args()
	
	encryptor = args.encoder
	device = torch.device('cuda')
	train_set = ppmi_pairs(mode = 'train')
	val_set = ppmi_pairs(mode = 'val')
	
	pos_train, neg_train = compute_ms_ssim(train_set, encryptor, device) 
	pos_val, neg_val = compute_ms_ssim(val_set, encryptor, device)
	
	thres, acc_train = find_ms_ssim_threshold(pos_train, neg_train)
		
	acc_val, TP, FP, TN, FN = classify_ms_ssim(pos_val, neg_val, thres)
	print('|- Model:{}'.format(encryptor))
	print('|- Train | threshold = {:.4f}, acc = {:.4f}'.format(thres, acc_train))
	print('|- Val | acc = {:.4f}, TP:{}, FP;{}, TN;{}, FN:{}'.format(acc_val, TP, FP, TN, FN))

	plot_dist(pos_train, neg_train)

if __name__ == '__main__':
	main()
