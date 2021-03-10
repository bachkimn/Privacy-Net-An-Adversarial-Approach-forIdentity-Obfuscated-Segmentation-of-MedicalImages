import torch
import torch.nn as nn
from densenet import densenet121 as Densenet
from resnet import resnet18 as Resnet

class discriminator(nn.Module):
	def __init__(self, arch_cnn='densenet'):
		super(discriminator, self).__init__()

		if arch_cnn.lower() == 'densenet':
			self.cnn_backbone = Densenet()
			print('|------Initiating Discriminator: Densenet ------|')
		if arch_cnn.lower() == 'resnet':
			self.cnn_backbone = Resnet()
			print('|------Initiating Discriminator: Resnet ------|')
		
		self.l2_norm = l2_normalize_embedding()
			
		self.classifier = nn.Sequential(nn.Linear(2000,4000, bias=True),
										nn.ReLU(),
										nn.Linear(4000,2,bias=True)
										)

		self.classifier_2 = nn.Sequential(nn.Linear(1000,2000, bias=True),
										nn.ReLU(),
										nn.Linear(2000,27,bias=True)
										)
										
	def extract_ft(self,x):
		ft = self.cnn_backbone(x)
		ft = self.l2_norm(ft)
		return ft

	def clf(self, ft1, ft2):
		ft = torch.cat([ft1,ft2],dim=1)
		pred = self.classifier(ft)	
		return pred
	
	def clf_patch(self, ft):
		pred = self.classifier_2(ft)
		return pred
			
	def forward(self,x1,x2):
		ft1 = self.extract_ft(x1)
		ft2 = self.extract_ft(x2) 
		pred = self.clf(ft1,ft2)
		#patch1 = self.clf_patch(ft1)
		#patch2 = self.clf_patch(ft2)
		return pred

class l2_normalize_embedding(nn.Module):
    def __init__(self):
        super(l2_normalize_embedding, self).__init__()
        
    def forward(self, x):
        return nn.functional.normalize(x)
