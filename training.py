import os 
import random
from pathlib import Path
import os
from datetime import datetime
from shutil import copyfile
import time

#Training Parameters
training_ratio = 0.7
outputFolder = r"D:\darknet\training"
directories = [
	r"D:\dataset\360Pedestrian\360Ped\Dataset",
	r"D:\dataset\360Pedestrian\task_2021.09.22-2021_09_24_03_32_18-yolo 1.1\obj_train_data"
]

#CFG Parameters
batch=64
subdivisions=32
classes=80
width=608
height=608


#Darknet Parameters
TRAINING_CFG_TEMPLATE = r"D:\darknet\darknetv3\build\darknet\x64\cfg\yolov3-tiny.cfg"
PretrainedWeight=r'D:\darknet\darknetv3\build\darknet\x64\yolov3-tiny.conv.15'
DARKNET_EXECUTABLE = r"D:\darknet\darknetv3\build\darknet\x64\darknet.exe"
RENAME_SCHEMA = "yolov3-%YYYY%MM%DD.cfg"
ACCEPT_IMAGES_TYPE = [".jpeg", ".png", ".jpg", ".PNG"]

#Create output folder using todate date
outputFolder = os.path.join(Path(outputFolder),f"{datetime.today().strftime('%Y')}-{datetime.today().strftime('%m')}-{datetime.today().strftime('%d')}")
if not os.path.exists(outputFolder):
    os.mkdir(outputFolder)


assert width%32==0 and height%32==0, 'width and height value must be the multiple of 32'
assert training_ratio>0 and training_ratio<1, 'TRAINING_RATIO should between 0 to 1'
assert batch%subdivisions == 0, 'batch should the multiple of subdivisions'

#Validate dataset
dataset = []
for directory in directories:
	files = [f for f in os.listdir(directory) if f.endswith(tuple(ACCEPT_IMAGES_TYPE)) ]

	valid_files = []
	for f in files:
		imageExtension = os.path.splitext(f)[1]
		textFilename = os.path.splitext(f)[0]
		textFile= textFilename+".txt"
		if os.path.exists(os.path.join(directory,textFile)):	
			valid_files.append(os.path.join(directory,f))
	dataset.extend(valid_files) 
	print(f"Find: {str(len(valid_files))}/{str(len(files))} images in {directory}")
	
assert len(dataset)>100, 'Dataset must have more than 100 images'

#Shuffle dataset
print(f"Total: {str(len(dataset))} images")
random.shuffle(dataset)
cutOffIdx = round(len(dataset) * training_ratio)
print(f"Dataset(Training:Testing): {str(cutOffIdx)} : {str(len(dataset)-cutOffIdx)} ")



max_batches=classes*2000 #max_batches: classes*2000, but not less than number of training images and not less than 6000
steps=f"{int(max_batches*.8)},{int(max_batches*.9)}" #max_batches*.8, max_batches*.9
#filters=(classes+5)*3 #if yolo: filters=(classes + 5)x3 elif Gaussian_yolo filters=(classes + 9)x3
if cutOffIdx>max_batches:
	max_batches = int(cutOffIdx * 2)
	steps=f"{int(max_batches*.8)},{int(max_batches*.9)}"

#Read and update network config
cfg = []
with open(TRAINING_CFG_TEMPLATE) as f:
	line = f.readlines()
	cfg = [f for f in line if not f.startswith('#') ]

sectionname = ""
lastFilterLine = 0
for no, line in enumerate(cfg):
	if line.startswith("["):
		sectionname = line
		if sectionname.startswith("[yolo]"):
			cfg[lastFilterLine] = f"filters = {(classes + 5) * 3}\n"
		elif sectionname.startswith("[Gaussian_yolo]"):
			cfg[lastFilterLine] = f"filters = {(classes + 9) * 3}\n"
	elif sectionname.startswith("[net]") and line.startswith("batch"):
		cfg[no] = f"batch = {batch}\n"
	elif sectionname.startswith("[net]") and line.startswith("subdivisions"):
		cfg[no] = f"subdivisions = {subdivisions}\n"
	elif line.startswith("classes"):
		cfg[no] = f"classes = {classes}\n"
	elif sectionname.startswith("[net]") and line.startswith("width"):
		cfg[no] = f"width = {width}\n"
	elif sectionname.startswith("[net]") and line.startswith("height"):
		cfg[no] = f"height = {height}\n"
	elif sectionname.startswith("[net]") and line.startswith("max_batches"):
		cfg[no] = f"max_batches = {max_batches}\n"
	elif sectionname.startswith("[net]") and line.startswith("steps"):
		cfg[no] = f"steps = {steps}\n"
	elif line.startswith("filters"):
		lastFilterLine = no

RENAME_SCHEMA = RENAME_SCHEMA.replace("%YYYY",datetime.today().strftime('%Y')).replace("%MM",datetime.today().strftime('%m')).replace("%DD",datetime.today().strftime('%d'))

#Write network config
NEW_CFG = os.path.join(Path(outputFolder),RENAME_SCHEMA)
copyfile(TRAINING_CFG_TEMPLATE, NEW_CFG)
f= open(NEW_CFG,"w+")
f.write(''.join(cfg))
f.close()

#Copy Pretrained Convolutional Weights to output directory
copyfile(PretrainedWeight, os.path.join(outputFolder,'yolov3-tiny.conv.15'))

#Create folder structure
if not os.path.exists(os.path.join(outputFolder,'data')):
    os.mkdir(os.path.join(outputFolder,'data'))
if not os.path.exists(os.path.join(outputFolder,'backup')):
    os.mkdir(os.path.join(outputFolder,'backup'))

#Write training dataset
trainingFiles = os.path.join(outputFolder,"data","train.txt")
print(f"Writing: {str(trainingFiles)}")
f= open(trainingFiles,"w+")
f.write('\n'.join(dataset[0:cutOffIdx]))
f.close()

#Write testing dataset
testingFiles = os.path.join(outputFolder,"data","test.txt")
print(f"Writing: {str(testingFiles)}")
f= open(testingFiles,"w+")
f.write('\n'.join(dataset[cutOffIdx:len(dataset)]))
f.close()

#Generate fake data
newDataContent = f"classes= {classes}\n" \
				f"train  = {trainingFiles}\n" \
				f"valid  = {testingFiles}\n" \
				f"names = {os.path.join(outputFolder,'data','setting.names')}\n" \
				f"backup = {os.path.join(outputFolder,'backup/')}\n"
f= open(os.path.join(outputFolder,'data','setting.data'),"w+")
f.write(newDataContent)
f.close()

#Generate fake classes.name
tempNames = ['N/A'] * classes
f= open(os.path.join(outputFolder,'data','setting.names'),"w+")
f.write('\n'.join(tempNames))
f.close()

os.chdir(Path(outputFolder))
cmd = f"{DARKNET_EXECUTABLE} detector train {os.path.join(outputFolder,'data','setting.data')} {os.path.join(outputFolder,RENAME_SCHEMA)} {os.path.join(outputFolder,'yolov3-tiny.conv.15')} "
print("Execute: "+cmd)

#Start training
start = time.time()
os.system(cmd)
end = time.time()
