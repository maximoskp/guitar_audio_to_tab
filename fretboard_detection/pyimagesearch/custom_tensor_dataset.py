# import the necessary packages
from torch.utils.data import Dataset
import torch 
import numpy as np

class CustomTensorDataset(Dataset):
	# initialize the constructor
	def __init__(self, tensors, transforms=None, augment_transforms=None):
		self.tensors = tensors
		self.transforms = transforms
		self.augment_transforms = augment_transforms

	def __getitem__(self, index):
		# grab the image, label, and its bounding box coordinates
		image = self.tensors[0][index]
		label = self.tensors[1][index]
		bbox = self.tensors[2][index]

		# __gbastas__
		image_prev = image.clone().detach() 
		bbox_prev = bbox.clone().detach() 
		if self.augment_transforms:
			bbox = self.from_yolo_to_standard(bbox)
			# print(np.array(image), np.array(bbox))
			image, bbox = self.augment_transforms(np.array(image), np.array(bbox[None,:]))
			# sometimes augmentation fails and we just keep the standard training sample
			try:
				# print(image.shape)
				# print(bbox[0].shape)
				image = torch.tensor(image)
				bbox = torch.tensor(bbox[0])
				bbox = self.from_standard_to_yolo(bbox)
			except IndexError as e:
				print(e)
				image = image_prev
				bbox = bbox_prev


		# transpose the image such that its channel dimension becomes
		# the leading one
		image = image.permute(2, 0, 1)

		# check to see if we have any image transformations to apply
		# and if so, apply them

		if self.transforms:
			image = self.transforms(image)
		# return a tuple of the images, labels, and bounding
		# box coordinates
		return (image, label, bbox)

	def __len__(self):
		# return the size of the dataset
		return self.tensors[0].size(0)

	def from_yolo_to_standard(self, bbox):
		# print(bbox[0])
		bbox[0] = bbox[0] - bbox[2]/2
		bbox[1] = bbox[1]- bbox[3]/2
		bbox[2] = bbox[0]+ bbox[2]/2
		bbox[3] = bbox[1]+ bbox[3]/2
		return bbox

	def from_standard_to_yolo(self, bbox):
		bbox[0] = (bbox[0] + bbox[2])/2
		bbox[1] = (bbox[1] + bbox[3])/2
		bbox[2] = bbox[2] - bbox[0]
		bbox[3] = bbox[3] - bbox[1]
		return bbox