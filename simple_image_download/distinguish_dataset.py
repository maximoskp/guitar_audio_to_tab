import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('-itype','--imgtype', type=str, default='.jpeg')
parser.add_argument('-ldir','--labels_dir', type=str, default='guitar-labels/labels-guitar-solo_2022-06-29-12-09-55')

args = parser.parse_args()

workdir = 'my_dir/'

instrument = args.labels_dir.split(os.sep)[-1].split('_')[1].split('-')[0]
images_dir = os.path.join(workdir,instrument+'-images/',args.labels_dir.split(os.sep)[-1].split('_')[1])

for label_filename in os.listdir(args.labels_dir):
    

    filestem = label_filename[::-1].split('.',1)[1][::-1]
    try:
        shutil.copy(os.path.join(images_dir, filestem+args.imgtype), workdir+instrument+'-dataset/')
    except FileNotFoundError as e:
        print('[gb] ' + os.path.join(images_dir, filestem+args.imgtype) + ' Image File Not Found!' )
        continue

    shutil.copy(os.path.join(args.labels_dir, label_filename), workdir+instrument+'-dataset/')


        