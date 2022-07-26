# import the necessary packages
from torch.nn import Dropout
from torch.nn import Identity
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Sequential
from torch.nn import Sigmoid
from torch import nn
import torch.nn.functional as F
import torch

class ObjectDetector(Module):
	def __init__(self, baseModel, numClasses):
		super(ObjectDetector, self).__init__()

		# initialize the base model and the number of classes
		self.baseModel = baseModel
		self.numClasses = numClasses

		# build the regressor head for outputting the bounding box
		# coordinates
		self.regressor = Sequential(
			# Linear(baseModel.fc.in_features, 128), # this is for ResNet
			Linear(baseModel.last_channel, 128), # this is for MobileNet
			ReLU(),
			Linear(128, 64),
			ReLU(),
			Linear(64, 32),
			ReLU(),
			Linear(32, 4),
			Sigmoid()
		)

	   # build the classifier head to predict the class labels
		# self.classifier = Sequential(
		# 	Linear(baseModel.fc.in_features, 512),
		# 	# Linear(baseModel.last_channel, 128),
		# 	ReLU(),
		# 	Dropout(),
		# 	Linear(512, 512),
		# 	ReLU(),
		# 	Dropout(),
		# 	Linear(512, self.numClasses)
		# )
		# set the classifier of our base model to produce outputs
		# from the last convolution block
		self.baseModel.fc = Identity()  # this is for ResNet
		self.baseModel.classifier = Identity() # this is for MobileNet


	def forward(self, x):
			# pass the inputs through the base model and then obtain
			# predictions from two different branches of the network
			features = self.baseModel(x)
			bboxes = self.regressor(features)
			# classLogits = self.classifier(features)
			classLogits = None # NOTE: __gbastas__ by-pass, maybe not important

			# return the outputs as a tuple
			return (bboxes, classLogits)



class ObjectDetector2(nn.Module):
	def __init__(self, baseModel, numClasses):
		super(ObjectDetector, self).__init__()

		# CNNs for rgb images
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
		self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
		self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5)
		self.conv4 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=5)
		self.conv5 = nn.Conv2d(in_channels=48, out_channels=192, kernel_size=5)


		#    # Connecting CNN outputs with Fully Connected layers for classification
		#    self.class_fc1 = nn.Linear(in_features=1728, out_features=240)
		#    self.class_fc2 = nn.Linear(in_features=240, out_features=120)
		#    self.class_out = nn.Linear(in_features=120, out_features=2)

		# Connecting CNN outputs with Fully Connected layers for bounding box
		self.box_fc1 = nn.Linear(in_features=768, out_features=240)
		self.box_fc2 = nn.Linear(in_features=240, out_features=120)
		self.box_out = nn.Linear(in_features=120, out_features=4)


	def forward(self, t):
		t = self.conv1(t)
		t = F.relu(t)
		t = F.max_pool2d(t, kernel_size=2, stride=2)

		t = self.conv2(t)
		t = F.relu(t)
		t = F.max_pool2d(t, kernel_size=2, stride=2)

		t = self.conv3(t)
		t = F.relu(t)
		t = F.max_pool2d(t, kernel_size=2, stride=2)

		t = self.conv4(t)
		t = F.relu(t)
		t = F.max_pool2d(t, kernel_size=2, stride=2)

		t = self.conv5(t)
		t = F.relu(t)
		t = F.avg_pool2d(t, kernel_size=4, stride=2)

		t = torch.flatten(t,start_dim=1)
		

		#    class_t = self.class_fc1(t)
		#    class_t = F.relu(class_t)

		#    class_t = self.class_fc2(class_t)
		#    class_t = F.relu(class_t)

		#    class_t = F.softmax(self.class_out(class_t),dim=1)

		# print('AAAAA', t.shape)

		box_t = self.box_fc1(t)
		box_t = F.relu(box_t)

		box_t = self.box_fc2(box_t)
		box_t = F.relu(box_t)

		box_t = self.box_out(box_t)
		box_t = F.sigmoid(box_t)

		return (box_t,None)
