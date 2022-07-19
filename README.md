# guitar_audio_to_tab

Install:

```
conda create --name av-guit python=3.7
conda activate av-guit
pip install opencv-python mediapipe protobuf==3.20.* matplotlib scipy numpy PyGuitarPro librosa tensorflow pickle5 torch torchvision opencv-contrib-python tqdm imutils
conda install -c anaconda pyaudio
```

Run:
```
python run_06_realtime.py -nn
```