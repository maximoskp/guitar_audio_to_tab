# import the necessary packages
from torch.utils.data import Dataset
import torch 
import numpy as np

import cv2


def write_image_with_bbox(orig, bbox): #[0,255]

	# orig = np.array(image).transpose(1,2,0)
	(centerX, centerY, widthX, heightY) = bbox
	(startX, startY, endX, endY) = (centerX - widthX/2, centerY - heightY/2, centerX + widthX/2, centerY + heightY/2)
	# (startX, startY, endX, endY) = (centerX, centerY, widthX, heightY)

	orig = imutils.resize(orig, width=600)
	(h, w) = orig.shape[:2]

	startX = int(startX * w)
	startY = int(startY * h)
	endX = int(endX * w)
	endY = int(endY * h)
	y = startY - 10 if startY - 10 > 10 else startY + 10
	cv2.putText(orig, 'neck', (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (0, 255, 0), 2)
	cv2.rectangle(orig, (startX, startY), (endX, endY),
		(0, 255, 0), 2)
	orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
	cv2.imwrite('check.jpeg', 255*orig)	


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
			image, bbox = self.augment_transforms(255*np.array(image), np.array(bbox[None,:]))
		
			# [gb] sometimes augmentation fails and we just keep the standard training sample
			try:
				image = torch.tensor(image)
				bbox = torch.tensor(bbox[0])
				bbox = self.from_standard_to_yolo(bbox)
			except IndexError as e:
				print(e)
				image = image_prev
				bbox = bbox_prev
				# write_image_with_bbox(image, bbox)

			
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
		startX = bbox[0] - bbox[2]/2
		startY = bbox[1] - bbox[3]/2
		endX = bbox[0] + bbox[2]/2
		endY = bbox[1] + bbox[3]/2
		bbox[0], bbox[1], bbox[2], bbox[3] = startX, startY, endX, endY
		return bbox

	def from_standard_to_yolo(self, bbox):
		centerX = (bbox[0] + bbox[2])/2
		centerY = (bbox[1] + bbox[3])/2
		widthX = bbox[2] - bbox[0]
		heightY = bbox[3] - bbox[1]
		bbox[0], bbox[1], bbox[2], bbox[3] = centerX, centerY, widthX, heightY
		return bbox
