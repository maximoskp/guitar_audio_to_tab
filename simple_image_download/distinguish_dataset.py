import argparse
import os
import shutil

parser = argparse.ArgumentParser()
# parser.add_argument('--images', type=str)
parser.add_argument('--labels', type=str)
# parser.add_argument('-o', '--output_dir', type=str)
args = parser.parse_args()

workdir = 'my_dir/'

images_dir = os.path.join(workdir,'guitar-images/',args.labels.split(os.sep)[-1].split('_')[1])

for label_filename in os.listdir(args.labels):
    

    filestem = label_filename[::-1].split('.',1)[1][::-1]
    try:
        shutil.copy(os.path.join(images_dir, filestem+'.jpeg'), workdir+'dataset/')
    except FileNotFoundError as e:
        print('[gb] ' + os.path.join(images_dir, filestem+'.jpeg') + ' Image File Not Found!' )
        continue

    shutil.copy(os.path.join(args.labels, label_filename), workdir+'dataset/')

    # with open(os.path.join(args.images, filename), "r") as f:

        