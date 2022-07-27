# guitar_audio_to_tab


## Fretboard Visual Detection:
Check REAMD.md in ```./freboard_detection```.


## Install:
```
conda create --name av-guit python=3.7
conda activate av-guit
pip install opencv-python mediapipe matplotlib scipy numpy cython PyGuitarPro librosa tensorflow pickle5 torch torchvision opencv-contrib-python tqdm imutils protobuf==3.20.*
conda install -c anaconda pyaudio
```

<!-- ln -s ./fretboard_detection/pyimagesearch ./pyimagesearch -->

In Linux, you may also need to run: 
```
sudo apt-get install -y python-all-dev
sudo apt-get install -y python3-pyaudio 
apt-get install -y apulse
apt-get install -y alsa-utils
```

## Download required files (NOTE: these might change -- written in 19/07/22)
- download https://imisathena-my.sharepoint.com/:u:/g/personal/g_bastas_athenarc_gr/EfBudL_V7JlNibEcZF7IPFsBkOaHif4xZAetdrGjVPshKg?e=GTYk8N and save to fretboard_detection/output
- download http://maxim.mus.auth.gr:3161/models/hand/tab_hand_full_CNN_out_current_best.hdf5 and save to models/hand/
- TODO: where to download pickles

## Run:
```
python run_06_realtime.py -nn
```
or 
```
python run_07_multiThreadRealtime.py
```