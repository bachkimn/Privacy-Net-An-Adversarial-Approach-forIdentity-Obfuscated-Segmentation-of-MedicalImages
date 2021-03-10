import os
import torch
import numpy as np
import nibabel as nib
from random import randint, choice
from torch.utils.data import Dataset, DataLoader
from ultils import *

class ppmi_pairs(Dataset):
	def __init__(self, mode = 'train', ratio = 0.5):
		self.mode = mode
		self.ratio = ratio
		
		if self.mode == 'train':
			self.data_folder = 'your data folder'
		elif self.mode == 'val':
			self.data_folder = 'your data folder'
		
		self.im_list = os.listdir(self.data_folder)
		self.positives, self.negatives = get_examples(self.im_list)
		self.patches = gen_crop_point()
		
		print('|- Loaded:"{}" as {} set'.format(self.data_folder, self.mode))
		print('|-> No of positives: {}, No of negatives:{}'.format(len(self.positives),len(self.negatives)))

	def __len__(self):
		return int(len(self.positives)/self.ratio)
	
	def __get_pos_item__(self, index):
		d = 1
		point = choice(self.patches)
		im, im_ref = self.positives[index]
		
		x = load_nii_to_tensor(os.path.join(self.data_folder,im,'T1.nii.gz'), point)
		x_ref = load_nii_to_tensor(os.path.join(self.data_folder,im_ref,'T1.nii.gz'), point)
		
		y = load_segmap_to_tensor(os.path.join(self.data_folder,im,'segmap.nii.gz'), point)
		y_ref = load_segmap_to_tensor(os.path.join(self.data_folder,im_ref,'segmap.nii.gz'), point)
		
		return	x, y, x_ref, y_ref, d, im, im_ref
	
	def __get_neg_item__(self, index):
		d = 0
		point = choice(self.patches)
		if self.mode.lower() == 'train':
			im, im_ref = choice(self.negatives)
		else:
			im, im_ref = self.negatives[index]
			
		x = load_nii_to_tensor(os.path.join(self.data_folder,im,'T1.nii.gz'), point)
		x_ref = load_nii_to_tensor(os.path.join(self.data_folder,im_ref,'T1.nii.gz'), point)
		
		y = load_segmap_to_tensor(os.path.join(self.data_folder,im,'segmap.nii.gz'), point)
		y_ref = load_segmap_to_tensor(os.path.join(self.data_folder,im_ref,'segmap.nii.gz'), point)
		
		return x, y, x_ref, y_ref, d, im, im_ref
				
	def __getitem__(self, index):
		
		if index < len(self.positives):
			idx = index
			x, y, x_ref, y_ref, d, im, im_ref = self.__get_pos_item__(idx)
			
		else:
			idx = index - len(self.positives)
			x, y, x_ref, y_ref, d, im, im_ref = self.__get_neg_item__(idx)
		
		d_p = 1
		d_n = 0
		#print(label,patch_no,idx,index,im1,im2)
		return x, x_ref, y, y_ref,d, d_p, d_n, im, im_ref

def gen_crop_point():
	points = []
	msk = nib.load('a brain mask').get_data()
	for x in range(32,144-32):
		for y in range(32,192-32):
			for z in range(32,160-32):
				if msk[x,y,z] == 0:
					pass
				else:
					x_ = x-32
					y_ = y-32
					z_ = z-32
					points += [(x_,y_,z_)]
	return points
