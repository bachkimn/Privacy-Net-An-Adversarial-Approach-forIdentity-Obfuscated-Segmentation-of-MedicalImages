import os
import nibabel as nib
import numpy as np
import torch
import matplotlib.pyplot as plt

def load_nii_to_tensor(filename, crop_point=None, size=64):
	im = nib.load(filename).get_fdata()
	im = (im - im.min()) / (im.max() - im.min())
	if crop_point != None:
		x,y,z = crop_point
		im = im[x:x+size, y:y+size, z:z+size]
	im_tensor = torch.from_numpy(im).float().view(-1, im.shape[0], im.shape[1], im.shape[2])
	return im_tensor

def load_nii_to_numpy(filename, crop_point=None, size=64):
	im = nib.load(filename).get_fdata()
	im = (im - im.min()) / (im.max() - im.min())
	if crop_point != None:
		x,y,z = crop_point
		im = im[x:x+size, y:y+size, z:z+size]
	return im

def get_subject_list(image_list):
	subject_list = {}
	for image in image_list:
		subject = image.split('_')[0]
		if subject in subject_list.keys():
			subject_list[subject].append(image)
		else:
			subject_list[subject] = [image]
	return subject_list

def get_examples(im_list):
	positives = []
	negatives = []
	for i in range(len(im_list)):
		for j in range(i+1,len(im_list)):
			subject1 = im_list[i].split('_')[0]
			subject2 = im_list[j].split('_')[0]
			if subject1 == subject2:
				positives += [(im_list[i],im_list[j])]
			else:
				negatives += [(im_list[i],im_list[j])]
	return positives, negatives

def to_onehot_numpy(x, num_classes):
	x = np.array(x,dtype = np.int)
	onehot = []
	for i in range(num_classes+1):
		_ = np.zeros(x.shape)
		mat = np.where(x==i)
		_[mat] = 1
		onehot.append(_)
	onehot = np.array(onehot) 
	return onehot

def onehot_tensor_to_segmap_numpy(onehot):
	onehot = onehot.detach().cpu().numpy()
	segmap = np.zeros(onehot[0,:,:,:].shape)
	for i in range(1,onehot.shape[0]):
		segmap[np.where(onehot[i,:,:,:] == 1)] = i
	return segmap
	
def load_segmap_to_tensor(filename, crop_point=None, size=64):
	segmap = np.around(nib.load(filename).get_data())
	segmap = to_onehot_numpy(segmap,num_classes=5)
	if crop_point !=None:
		x,y,z = crop_point
		segmap = segmap[:,x:x+size, y:y+size, z:z+size]
	
	segmap_tensor = torch.from_numpy(segmap).float()
	return segmap_tensor

def plot_curves(x1, y1, x2, y2, save_dir, title, epoch):
	if os.path.exists(os.path.join(save_dir, title)) == False:
		os.mkdir(os.path.join(save_dir, title))
	plt.plot(x1, y1, x2, y2)
	plt.title(title)
	plt.savefig(os.path.join(save_dir, title, str(epoch) + '.png'))
	plt.clf()
	plt.close()
	return
