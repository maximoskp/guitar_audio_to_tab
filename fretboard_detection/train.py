# USAGE
# python train.py
# import the necessary packages
from pyimagesearch.bbox_regressor import ObjectDetector
from pyimagesearch.custom_tensor_dataset import CustomTensorDataset
from pyimagesearch import config
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
from torch.optim import Adam
from torchvision.models import resnet50
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
# aaaa
# define normalization transforms
transforms = transforms.Compose([
	transforms.ToPILImage(),
	transforms.ToTensor(),
	transforms.Normalize(mean=MEAN, std=STD)
])

transforms = transforms.Compose([
	transforms.ToPILImage(),
	transforms.ToTensor(),
	transforms.RandomAffine(degrees=(-10,30), translate=(0.1, 0.3), scale=(0.5, 0.75))
])


# convert NumPy arrays to PyTorch datasets
trainDS = CustomTensorDataset((trainImages, trainLabels, trainBBoxes),
	transforms=transforms)
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
	# initialize the number of correct predictions in the training
	# and validation step
	trainCorrect = 0
	valCorrect = 0
	# loop over the training set
	for (images, labels, bboxes) in trainLoader:
		# send the input to the device
		(images, labels, bboxes) = (images.to(config.DEVICE), labels.to(config.DEVICE), bboxes.to(config.DEVICE))
		# perform a forward pass and calculate the training loss
		predictions = objectDetector(images)
		bboxLoss = bboxLossFunc(predictions[0], bboxes)
		totalLoss = bboxLoss
		# classLoss = classLossFunc(predictions[1], labels)
		# totalLoss = (config.BBOX * bboxLoss) + (config.LABELS * classLoss)

		opt.zero_grad()
		totalLoss.backward()
		opt.step()

		totalTrainLoss += totalLoss
		# trainCorrect += (predictions[1].argmax(1) == labels).type(torch.float).sum().item()    

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
			# classLoss = classLossFunc(predictions[1], labels)
			# totalLoss = (config.BBOX * bboxLoss) + (config.LABELS * classLoss)
			totalValLoss += totalLoss
			# calculate the number of correct predictions
			# valCorrect += (predictions[1].argmax(1) == labels).type(torch.float).sum().item()
	# calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgValLoss = totalValLoss / valSteps
	# calculate the training and validation accuracy
	trainCorrect = trainCorrect / len(trainDS)
	valCorrect = valCorrect / len(testDS)
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