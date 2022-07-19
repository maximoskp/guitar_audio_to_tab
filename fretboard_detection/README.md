`<!-- https://pyimagesearch.com/2021/11/01/training-an-object-detector-from-scratch-in-pytorch/?utm_source=pocket_mylist -->

## Move data where they should be
mkdir -p dataset/images/neck/
mkdir dataset/annotations
cp ../simple_image_download/my_dir/dataset/*txt dataset/annotations/
cp ../simple_image_download/my_dir/dataset/*jpeg dataset/images/neck/


## Download required files (NOTE: these might change -- written in 19/07/22)
- download https://imisathena-my.sharepoint.com/:u:/g/personal/g_bastas_athenarc_gr/EfBudL_V7JlNibEcZF7IPFsBkOaHif4xZAetdrGjVPshKg?e=GTYk8N and save to fretboard_detection/output
- download http://maxim.mus.auth.gr:3161/models/hand/tab_hand_full_CNN_out_current_best.hdf5 and save to models/hand/
- TODO: where to download pickles

## TRAINING
python train.py

## INFERENCE
python predict.py -i dataset/images/neck/guitar-bill-frisell-solo_105.jpeg
python predict.py -i dataset/images/neck/guitar-bill-frisell-solo_105.jpeg
python predict.py -i dataset/images/neck/guitar-bill-frisell-solo_105.jpeg
python predict.py -i dataset/images/neck/guitar-bill-frisell-solo_3ab.jpeg
python predict.py -i dataset/images/neck/guitar-bill-frisell-solo_1c.jpeg