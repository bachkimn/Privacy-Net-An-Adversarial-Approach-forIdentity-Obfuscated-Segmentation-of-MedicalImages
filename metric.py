import numpy as np

def count_predictions(pred, target):
	pred = pred.argmax(dim=1)
	TP = (pred * target).sum()
	FP = ((pred - target) == 1).sum()
	TN = ((pred==0) * (target==0)).sum()
	FN = (((pred==0) - (target==0))==1).sum()
	return TP.item(),FP.item(),TN.item(),FN.item()

def compute_metric(TP,FP,TN,FN):
	acc = (TP+TN) / (TP+FP+TN+FN)
	if TP !=0:
		precision = TP / (TP + FP)
		recall = TP / (TP + FN)
		F1 = 2 * precision * recall / (precision + recall)
	else:
		precision = 0
		recall = 0
		F1 = 0
	return precision, recall, acc, F1
	
def mtc_classification_acc(pred, target):
	pred = pred.argmax(dim=1)
	acc = (pred==target).sum().item()/len(target)
	return acc


def compute_dice_score(pred, labels):
	'''pred: tensor of [B,C,X,Y,Z]
	   labels: tensor of [B,C,X,Y,Z]'''
	dsc = []   
	for i in range(len(pred[0,:,0,0,0])):
		if i == 0:
			dsc.append((2 * (pred[:,1:,:,:,:] * labels[:,1:,:,:,:]).sum() / (pred[:,1:,:,:,:] + labels[:,1:,:,:,:]).sum()).item())
		else:
			dsc.append((2 * (pred[:,i,:,:,:] * labels[:,i,:,:,:]).sum() / (pred[:,i,:,:,:] + labels[:,i,:,:,:]).sum()).item())
	return np.array(dsc)
