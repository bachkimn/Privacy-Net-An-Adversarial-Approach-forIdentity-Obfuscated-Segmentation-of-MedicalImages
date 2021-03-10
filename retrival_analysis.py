import os
import argparse
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from time import time
from math import pi
from autoencoder import Unet3D_encoder as encoder
from discriminator import discriminator
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from torch.utils.data import Dataset, DataLoader

class ppmi(Dataset):
	def __init__(self, data_folder='your data folder'):
		self.data_folder = data_folder
		self.im_list = os.listdir(data_folder)
		print('||- Loaded:"{}" included:{} images'.format(self.data_folder,len(self.im_list)))

	def __len__(self):
		return len(self.im_list)

	def __getitem__(self, index):
		im = self.im_list[index]
		subject = im.split('_')[0]
		x = self.__load_nii_to_tensor__(os.path.join(self.data_folder, im, 'T1.nii.gz'))
		return x, subject, im

	def __load_nii_to_tensor__(self, filename):
		im = nib.load(filename).get_fdata()
		im = (im - im.min()) / (im.max() - im.min())
		im_tensor = torch.from_numpy(im).float().view(-1, im.shape[0], im.shape[1], im.shape[2])
		return im_tensor

def extract_features(models, dataset, device):
	data_loader = DataLoader(dataset, batch_size = 1, shuffle = False)
	Encrypter = models['enc'].eval()
	Discriminator = models['dis'].eval()
	
	ft_vectors = []
	subject_list = []
	im_list = []
	print('Extract features')
	with torch.no_grad():
		for i, (x, subject, im) in enumerate(data_loader):
			x = x.to(device)
			z = Encrypter(x)
			ft = Discriminator.extract_ft(z)
			ft = ft.cpu().numpy()
			ft_vectors += list(ft)
			subject_list += subject
			im_list += im
			print('{}|{}|{}'.format(i,im,subject))
	ft_vectors = np.array(ft_vectors)
	return ft_vectors, subject_list, im_list
	
def get_dist(ft_vectors):
	print('Compute Cosine Distance')
	dist = np.arccos(1 - squareform(pdist(ft_vectors, metric = 'cosine')))/pi
	return dist

def get_rel(subject_list, im_list):
	'Generate Rel Matrix'
	rel = np.zeros((len(im_list), len(im_list)))
	for i in range(len(im_list)):
		for j in range(len(im_list)):
			subject_i = im_list[i].split('_')[0]
			subject_j = im_list[j].split('_')[0]
			if subject_i == subject_j:
				rel[i,j] = 1
				print('{}|{}|{}|{}|{}'.format(subject_i, subject_j, i, j, rel[i,j]))
	return rel

def get_subject_dict(image_list):
	subject_list = {}
	for image in image_list:
		subject = image.split('_')[0]
		if subject in subject_list.keys():
			subject_list[subject].append(image)
		else:
			subject_list[subject] = [image]
	return subject_list

def get_topk(im_name, distance_matrix, im_list, k):
	topk = []
	im_index = im_list.index(im_name)
	distance_array = distance_matrix[im_index,:]
	sorted_distance_index = np.argsort(distance_array)
	topk_idx = sorted_distance_index[1:1+k]
	for idx in topk_idx:
		topk += [im_list[idx]]
	return topk

def get_same_subject(im_name, subject_dict):
	subject = im_name.split('_')[0]
	same_subject = subject_dict[subject]
	return same_subject

def get_intersection(same_subject, topk):
	intersection = []
	for im in same_subject:
		if im in topk:
			intersection += [im]
	return len(intersection)

def compute_precision(im_name, k, distance_matrix, im_list, subject_dict):
	same_subject = get_same_subject(im_name, subject_dict)
	topk = get_topk(im_name, distance_matrix, im_list, k)
	precision = get_intersection(same_subject, topk) / k
	return precision

def compute_AvePi(im_name, n,distance_matrix, im_list, subject_dict, rel_matrix):
	same_subject = get_same_subject(im_name, subject_dict)
	precision = []
	i = im_list.index(im_name)
	for k in range(1,1+n):
		_topk = get_topk(im_name, distance_matrix, im_list, k)
		j = im_list.index(_topk[-1])
		rel_k = rel_matrix[i,j]
		_prec = compute_precision(im_name, k, distance_matrix, im_list, subject_dict) * rel_k
		precision += [_prec]
		
	precision = np.array(precision)
	AvePi = (1 / (len(same_subject) - 1)) * precision.sum() 
	return AvePi

def compute_MAP(distance_matrix, im_list, subject_dict, rel_matrix, n):
	AvePi = []
	for subject in subject_dict.keys():
		if len(subject_dict[subject]) >= 2:
			print('{}'.format(subject_dict[subject]))
			for im_name in subject_dict[subject]:
				_avepi = compute_AvePi(im_name, n,distance_matrix, im_list, subject_dict, rel_matrix)
				AvePi += [_avepi]
	AvePi = np.array(AvePi)
	MAP = AvePi.mean()
	return MAP
	


def args_parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--enc', type=str)
	parser.add_argument('--dis', type=str)
	
	args = parser.parse_args()
	return args

def main():
	args = args_parse()
	enc_path = args.enc
	dis_path = args.dis
	
	device = torch.device('cuda')
	
	# Init models
	Encoder = encoder(1,1,16).to(device)
	Encoder.load_state_dict(torch.load(enc_path, map_location=device))
	Encoder.eval()
	
	Discriminator = discriminator().to(device)
	Discriminator.load_state_dict(torch.load(dis_path, map_location=device))
	Discriminator.eval()
	
	models = {'enc': Encoder, 'dis': Discriminator}
	
	# Init data
	val_set = ppmi()
	
	# Extract features
	ft_vectors, subject_list, im_list = extract_features(models, val_set, device)
	distance_matrix = get_dist(ft_vectors)
	rel_matrix = get_rel(subject_list, im_list)
	# Compute similarity and rank similarity
	subject_dict = get_subject_dict(im_list)
	# Compute sensitivity, precision and d-prime
	MAP = compute_MAP(distance_matrix, im_list, subject_dict, rel_matrix, 5)
	print('MAP:{}'.format(MAP))
		
if __name__ == '__main__':
	main()

