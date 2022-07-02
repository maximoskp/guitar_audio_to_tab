<!-- https://pyimagesearch.com/2021/11/01/training-an-object-detector-from-scratch-in-pytorch/?utm_source=pocket_mylist -->

## Move data where they should be
cp ../../simple_image_download/my_dir/dataset/*txt annotations/
cp ../../simple_image_download/my_dir/dataset/*jpeg images/neck/

## TRAINING
python train.py

## INFERENCE
python predict.py -i dataset/images/neck/guitar-bill-frisell-solo_105.jpeg
python predict.py -i dataset/images/neck/guitar-bill-frisell-solo_105.jpeg
python predict.py -i dataset/images/neck/guitar-bill-frisell-solo_105.jpeg
python predict.py -i dataset/images/neck/guitar-bill-frisell_3ab.jpeg
python predict.py -i dataset/images/neck/guitar-bill-frisell_1c.jpeg