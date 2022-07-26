# USAGE
# python train.py
# import the necessary packages
from pyimagesearch.bbox_regressor import ObjectDetector
from pyimagesearch.custom_tensor_dataset import CustomTensorDataset
from pyimagesearch import config
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
from torch.optim import Adam
from torchvision.models import resnet50, mobilenet_v3_small
from sklearn.model_selection import train_test_split
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import time
import cv2
import os
import random
from data_aug import data_aug
import imutils
import warnings
warnings.filterwarnings("ignore")

def write_image_with_bbox(orig, bbox): #[0,255]

	# orig = np.array(image).transpose(1,2,0)
	(centerX, centerY, widthX, heightY) = bbox
	(startX, startY, endX, endY) = (centerX - widthX/2, centerY - heightY/2, centerX + widthX/2, centerY + heightY/2)

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



# https://blog.paperspace.com/data-augmentation-for-object-detection-building-input-pipelines/
class Sequence(object):
	def __init__(self, augmentations, probs = 1):

		
		self.augmentations = augmentations
		self.probs = probs

	def __call__(self, images, bboxes):
		for i, augmentation in enumerate(self.augmentations):
			if type(self.probs) == list:
				prob = self.probs[i]
			else:
				prob = self.probs
				
			if random.random() < prob:
				images, bboxes = augmentation(images, bboxes)
		return images, bboxes


# initialize the list of data (images), class labels, target bounding
# box coordinates, and image paths
print("[INFO] loading dataset...")
data = []
labels = []
bboxes = []
imagePaths = []

Labels = {'0':'neck'}

# YOLO style annotations 
for txtPath in paths.list_files(config.ANNOTS_PATH, validExts=(".txt")):
	filename = txtPath.split(os.sep)[-1][::-1].split('.',1)[1][::-1]+'.jpeg'
	rows = open(txtPath).read().strip().split("\n")
	# loop over the rows (even if we only have one image label involved: 'neck')
	for row in rows:
		row = row.split(" ")
		(labelID, centerX, centerY, widthX, heightY) = row # range [0,1]
		imagePath = os.path.sep.join([config.IMAGES_PATH, Labels[labelID], filename])
		image = cv2.imread(imagePath).astype(np.float32) / 255 # NOTE: range [0,1]
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = cv2.resize(image, (224, 224)) # no need to resize bboxes since they are set with relevant values [0,1]

		data.append(image)
		labels.append(Labels[labelID])
		bboxes.append((centerX, centerY, widthX, heightY))
		imagePaths.append(imagePath)

data = np.array(data, dtype="float32")
labels = np.array(labels)
bboxes = np.array(bboxes, dtype="float32")
imagePaths = np.array(imagePaths)
le = LabelEncoder()
labels = le.fit_transform(labels)

# def from_yolo_to_standard(bbox):
	# startX = bbox[0] - bbox[2]/2
	# startY = bbox[1] - bbox[3]/2
	# endX = bbox[0] + bbox[2]/2
	# endY = bbox[1] + bbox[3]/2
	# bbox[0], bbox[1], bbox[2], bbox[3] = startX, startY, endX, endY
	# return bbox


# def from_standard_to_yolo(bbox):
	# centerX = (bbox[0] + bbox[2])/2
	# centerY = (bbox[1] + bbox[3])/2
	# widthX = bbox[2] - bbox[0]
	# heightY = bbox[3] - bbox[1]
	# bbox[0], bbox[1], bbox[2], bbox[3] = centerX, centerY, widthX, heightY
	# return bbox


# augment_transforms = Sequence([data_aug.RandomScale(0.4, diff = False)])#, transforms.RandomScale(0.2, diff = True), transforms.RandomRotate(10)]))
# bbox = from_yolo_to_standard(bboxes[1])
# image = data[1]
# image, bbox = augment_transforms(255*data[1], np.array(bbox[None,:]))
# bbox = bbox[0]
# bbox = from_standard_to_yolo(bbox)
# write_image_with_bbox(image, bbox)
# aaaa

# partition the data into training and testing 
split = train_test_split(data, labels, bboxes, imagePaths,
	test_size=0.20, random_state=42)
# unpack the data split
(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]
(trainBBoxes, testBBoxes) = split[4:6]
(trainPaths, testPaths) = split[6:]

# convert NumPy arrays to PyTorch tensors
(trainImages, testImages) = torch.tensor(trainImages),\
	torch.tensor(testImages)
(trainLabels, testLabels) = torch.tensor(trainLabels),\
	torch.tensor(testLabels)
(trainBBoxes, testBBoxes) = torch.tensor(trainBBoxes),\
	torch.tensor(testBBoxes)


# __gbastas__
MEAN, STD = trainImages.mean([0,1,2]), trainImages.std([0,1,2])
print(MEAN)
print(STD)


# augment_transforms = None
# define normalization transforms
transforms = transforms.Compose([
	transforms.ToPILImage(),
	transforms.ToTensor(),
	transforms.Normalize(mean=MEAN, std=STD)
])

# augment_transforms = Sequence([data_aug.RandomScale(0.4, diff = False)])#, transforms.RandomScale(0.2, diff = True), transforms.RandomRotate(10)]))
augment_rescale = Sequence([data_aug.RandomScale((-0.4,0), diff = False)])
# augment_crop = Sequence([data_aug.RandomScale((0,0.4), diff = False)])#, transforms.RandomScale(0.2, diff = True), transforms.RandomRotate(10)]))
# augment_rotate = Sequence([data_aug.RandomRotate(10)])
# augment_shear = Sequence([data_aug.RandomShear()])
augment_hsv = Sequence([data_aug.RandomHSV(hue=(-80,80), saturation=(-80,80), brightness=(-80,80)), data_aug.RandomScale((0,0.4), diff = False)])

# convert NumPy arrays to PyTorch datasets
trainDS = CustomTensorDataset((trainImages, trainLabels, trainBBoxes),
	transforms=transforms)#, augment_transforms=augment_transforms)
trainDS_rescaled = CustomTensorDataset((trainImages, trainLabels, trainBBoxes),
	transforms=transforms, augment_transforms=augment_rescale)
# trainDS_cropped = CustomTensorDataset((trainImages, trainLabels, trainBBoxes),
# 	transforms=transforms, augment_transforms=augment_crop)	
# trainDS_rotate = CustomTensorDataset((trainImages, trainLabels, trainBBoxes),
# 	transforms=transforms, augment_transforms=augment_rotate)
# trainDS_shear = CustomTensorDataset((trainImages, trainLabels, trainBBoxes),
# 	transforms=transforms, augment_transforms=augment_shear)
trainDS_hsv = CustomTensorDataset((trainImages, trainLabels, trainBBoxes),
	transforms=transforms, augment_transforms=augment_hsv)

trainDS = ConcatDataset([trainDS, trainDS_rescaled])
# trainDS = ConcatDataset([trainDS, trainDS_cropped])
# trainDS = ConcatDataset([trainDS, trainDS_rotate])
# trainDS = ConcatDataset([trainDS, trainDS_shear])
trainDS = ConcatDataset([trainDS, trainDS_hsv])

testDS = CustomTensorDataset((testImages, testLabels, testBBoxes),
	transforms=transforms)
print("[INFO] total training samples: {}...".format(len(trainDS)))
print("[INFO] total test samples: {}...".format(len(testDS)))
# calculate steps per epoch for training and validation set
trainSteps = len(trainDS) // config.BATCH_SIZE
valSteps = len(testDS) // config.BATCH_SIZE
# create data loaders
trainLoader = DataLoader(trainDS, batch_size=config.BATCH_SIZE,
	shuffle=True, num_workers=os.cpu_count(), pin_memory=config.PIN_MEMORY)
testLoader = DataLoader(testDS, batch_size=config.BATCH_SIZE,
	num_workers=os.cpu_count(), pin_memory=config.PIN_MEMORY)



print("[INFO] saving testing image paths...")
f = open(config.TEST_PATHS, "w")
f.write("\n".join(testPaths))
f.close()
resnet = resnet50(pretrained=True)
# resnet = resnet50(pretrained=False)

for param in resnet.parameters():
	# param.requires_grad = False
	param.requires_grad = True

# create our custom object detector model and flash it to the current
# device
objectDetector = ObjectDetector(resnet, len(le.classes_))
objectDetector = objectDetector.to(config.DEVICE)
# define our loss functions
# classLossFunc = CrossEntropyLoss()
bboxLossFunc = MSELoss()
# initialize the optimizer, compile the model, and show the model
# summary
opt = Adam(objectDetector.parameters(), lr=config.INIT_LR)
print(objectDetector)
# initialize a dictionary to store training history
H = {"total_train_loss": [], "total_val_loss": [], "train_class_acc": [],
	 "val_class_acc": []}

# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
for e in tqdm(range(config.NUM_EPOCHS)):
	# set the model in training mode
	objectDetector.train()
	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalValLoss = 0
	# loop over the training set
	for (images, labels, bboxes) in trainLoader:
		if totalTrainLoss==0:
			orig = np.array(images[0]).transpose(1,2,0)
			write_image_with_bbox(orig, bboxes[0])

		# send the input to the device
		(images, labels, bboxes) = (images.to(config.DEVICE), labels.to(config.DEVICE), bboxes.to(config.DEVICE))
		# perform a forward pass and calculate the training loss
		predictions = objectDetector(images)
		bboxLoss = bboxLossFunc(predictions[0], bboxes)
		totalLoss = bboxLoss

		opt.zero_grad()
		totalLoss.backward()
		opt.step()

		totalTrainLoss += totalLoss

	# switch off autograd
	with torch.no_grad():
		# set the model in evaluation mode
		objectDetector.eval()
		# loop over the validation set
		for (images, labels, bboxes) in testLoader:
			# send the input to the device
			(images, labels, bboxes) = (images.to(config.DEVICE),
				labels.to(config.DEVICE), bboxes.to(config.DEVICE))
			# make the predictions and calculate the validation loss
			predictions = objectDetector(images)
			bboxLoss = bboxLossFunc(predictions[0], bboxes)
			totalLoss = bboxLoss
			totalValLoss += totalLoss
	# calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgValLoss = totalValLoss / valSteps

	# update our training history
	H["total_train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	# H["train_class_acc"].append(trainCorrect)
	H["total_val_loss"].append(avgValLoss.cpu().detach().numpy())
	# H["val_class_acc"].append(valCorrect)
	# print the model training and validation information
	print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
	print("Train loss: {:.6f}".format(
		avgTrainLoss))
	print("Val loss: {:.6f}".format(
		avgValLoss))

endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))                
# serialize the model to disk
print("[INFO] saving object detector model...")
torch.save(objectDetector, config.MODEL_PATH)
# serialize the label encoder to disk
print("[INFO] saving label encoder...")
f = open(config.LE_PATH, "wb")
f.write(pickle.dumps(le))
f.close()
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["total_train_loss"], label="total_train_loss")
plt.plot(H["total_val_loss"], label="total_val_loss")
# plt.plot(H["train_class_acc"], label="train_class_acc")
# plt.plot(H["val_class_acc"], label="val_class_acc")
plt.title("Total Training Loss and Classification Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
# save the training plot
plotPath = os.path.sep.join([config.PLOTS_PATH, "training.png"])
plt.savefig(plotPath)