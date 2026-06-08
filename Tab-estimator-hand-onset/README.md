python src/predict.py guitarset_guitartechs_egdb_handpos_clean/guitarset_guitartechs_egdb_handpos_onset 192 --test-num 0   --event-label-delay-ms 25   --event-label-window-ms 50   --event-string-window-ms 70 --onset-threshold 0.80


CUDA_VISIBLE_DEVICES=1 python src/train.py   --npz-dir data/npz/guitarset/split   --run-name guitarset_onset_large_gloabel_lowerpweights  --test-num 0   --epoch 192     --onset-target-radius-ms 25   --onset-target-mode triangular   --onset-loss-weight 0.25   --global-onset-loss-weight 0.5   --onset-positive-weight 5.0   --global-onset-positive-weight 5.0

python src/predict.py guitarset/guitarset_onset_large_gloabel_lowerpweights 192 --test-num 0   --event-label-delay-ms 25   --event-label-window-ms 50   --event-string-window-ms 70 --onset-threshold 0.80

CUDA_VISIBLE_DEVICES=2 python src/train.py   --npz-dir data/npz/guitarset_guitartechs_egdb_goat_idmt_handpos_clean/split   --run-name guitarset_guitartechs_egdb_goat_idmt_handpos_onset_III   --use-hand-position   --test-num 6  --n-folds 7 --epoch 192

[old]
python src/predict.py  guitarset_guitartechs_egdb_goat_idmt_handpos_clean/guitarset_guitartechs_egdb_goat_idmt_handpos_onset_III 192 --test-num 06   --event-label-delay-ms 50   --event-label-window-ms 50   --event-string-window-ms 70 --onset-threshold 0.80 --n-folds 7 --npz-dir data/npz/guitarset_guitartechs_egdb_goat_idmt_handpos_clean/split
frame_frame_avg_tab_f = 0.7039
frame_avg_onset_f     = 0.6314
event_avg_f           = 0.7693

[new]
python src/predict.py   guitarset_guitartechs_egdb_goat_idmt_handpos_clean/guitarset_guitartechs_egdb_goat_idmt_handpos_onset_III   192   --test-num 6   --n-folds 7   --npz-dir data/npz/guitarset_guitartechs_egdb_goat_idmt_handpos_clean/split   --event-decode-mode string_onset   --event-chord-group-ms 55   --event-label-delay-ms 20   --event-label-window-ms 90   --event-string-window-ms 50   --event-tab-threshold 0.30   --onset-threshold 0.80   --global-onset-threshold 0.80   --peak-pre-max-ms 30   --peak-post-max-ms 30   --peak-combine-ms 30   --use-global-onset-fallback   --global-fallback-max-notes 2   --repeat-same-fret-policy strong_onset   --min-repeat-ms 250   --repeat-onset-threshold 0.94   --repeat-global-threshold 0.75   --same-string-any-fret-min-ms 80
Test No. 06
Headline metrics
frame_frame_avg_tab_f = 0.7039
frame_avg_onset_f     = 0.6164
event_avg_f           = 0.7551


-----------------

python src/train.py \
  --npz-dir data/npz/guitarset_guitartechs_egdb_goat_idmt_handpos_clean/split \
  --run-name guitarset_guitartechs_egdb_goat_idmt_handpos_onset_ENCODER_ONLY_NO_HAND \
  --use-hand-position \
  --hand-position-fusion prior \
  --no-onset-raw-features \
  --test-num 6 \
  --n-folds 7 \
  --epoch 192

python src/predict.py  guitarset_guitartechs_egdb_goat_idmt_handpos_clean/guitarset_guitartechs_egdb_goat_idmt_handpos_onset_ENCODER_ONLY_NO_HAND 192 --test-num 06   --event-label-delay-ms 50   --event-label-window-ms 50   --event-string-window-ms 70 --onset-threshold 0.80 --n-folds 7 --npz-dir data/npz/guitarset_guitartechs_egdb_goat_idmt_handpos_clean/split
frame_frame_avg_tab_f = 0.5607
frame_avg_onset_f     = 0.5104
event_avg_f           = 0.6256

python src/train.py \
  --npz-dir data/npz/guitarset_guitartechs_egdb_goat_idmt_handpos_clean/split \
  --run-name guitarset_guitartechs_egdb_goat_idmt_handpos_onset_NO_RAW_ONSET \
  --use-hand-position \
  --hand-position-fusion hidden+prior \
  --no-onset-raw-features \
  --test-num 6 \
  --n-folds 7 \
  --epoch 192



python src/predict.py  guitarset_guitartechs_egdb_goat_idmt_handpos_clean/guitarset_guitartechs_egdb_goat_idmt_handpos_onset_NO_RAW_ONSET 192 --test-num 06   --event-label-delay-ms 50   --event-label-window-ms 50   --event-string-window-ms 70 --onset-threshold 0.80 --n-folds 7 --npz-dir data/npz/guitarset_guitartechs_egdb_goat_idmt_handpos_clean/split

frame_frame_avg_tab_f = 0.6301
frame_avg_onset_f     = 0.6216
event_avg_f           = 0.7539

python src/train.py \
  --npz-dir data/npz/guitarset_guitartechs_egdb_goat_idmt_handpos_clean/split \
  --run-name guitarset_guitartechs_egdb_goat_idmt_handpos_onset_GLOBAL_ONLY \
  --use-hand-position \
  --hand-position-fusion hidden+prior \
  --onset-loss-weight 0 \
  --global-onset-loss-weight 0.25 \
  --test-num 6 \
  --n-folds 7 \
  --epoch 192

python src/predict.py  guitarset_guitartechs_egdb_goat_idmt_handpos_clean/guitarset_guitartechs_egdb_goat_idmt_handpos_onset_GLOBAL_ONLY 192 --test-num 06   --event-label-delay-ms 50   --event-label-window-ms 50   --event-string-window-ms 70 --onset-threshold 0.80 --n-folds 7 --npz-dir data/npz/guitarset_guitartechs_egdb_goat_idmt_handpos_clean/split

Headline metrics
frame_frame_avg_tab_f = 0.6679
frame_avg_onset_f     = 0.0000
event_avg_f           = 0.7182n

python src/train.py \
  --npz-dir data/npz/guitarset_guitartechs_egdb_goat_idmt_handpos_clean/split \
  --run-name guitarset_guitartechs_egdb_goat_idmt_handpos_onset_STRING_ONLY \
  --use-hand-position \
  --hand-position-fusion hidden+prior \
  --onset-loss-weight 0.25 \
  --global-onset-loss-weight 0 \
  --test-num 6 \
  --n-folds 7 \
  --epoch 192

python src/predict.py  guitarset_guitartechs_egdb_goat_idmt_handpos_clean/guitarset_guitartechs_egdb_goat_idmt_handpos_onset_STRING_ONLY 192 --test-num 06   --event-label-delay-ms 50   --event-label-window-ms 50   --event-string-window-ms 70 --onset-threshold 0.80 --n-folds 7 --npz-dir data/npz/guitarset_guitartechs_egdb_goat_idmt_handpos_clean/split

Headline metrics
frame_frame_avg_tab_f = 0.6639
frame_avg_onset_f     = 0.6145
event_avg_f           = 0.1000


python src/train.py   --npz-dir data/npz/guitarset_guitartechs_egdb_goat_idmt/split   --run-name guitarset_guitartechs_egdb_goat_idmt_hanspos_onset_NO_RAW    --hand-position-fusion prior   --test-num 6   --n-folds 7   --epoch 192 --no-onset-raw-features

python src/predict.py  guitarset_guitartechs_egdb_goat_idmt/guitarset_guitartechs_egdb_goat_idmt_onset_NO_RAW 192 --test-num 06   --event-label-delay-ms 50   --event-label-window-ms 50   --event-string-window-ms 70 --onset-threshold 0.80 --n-folds 7 --npz-dir data/npz/guitarset_guitartechs_egdb_goat_idmt_handpos_clean/split


python src/train.py   --npz-dir data/npz/guitarset_guitartechs_egdb_goat_idmt_handpos_clean/split   --run-name guitarset_guitartechs_egdb_goat_idmt_handpos_onset_ENCODER_ONLY_NO_HAND_WITH_RAW   --use-hand-position   --hand-position-fusion prior   --test-num 6   --n-folds 7   --epoch 192

python src/predict.py  guitarset_guitartechs_egdb_goat_idmt_handpos_clean/guitarset_guitartechs_egdb_goat_idmt_handpos_onset_ENCODER_ONLY_NO_HAND_WITH_RAW 192 --test-num 06   --event-label-delay-ms 50   --event-label-window-ms 50   --event-string-window-ms 70 --onset-threshold 0.80 --n-folds 7 --npz-dir data/npz/guitarset_guitartechs_egdb_goat_idmt_handpos_clean/split
Headline metrics
frame_frame_avg_tab_f = 0.5824
frame_avg_onset_f     = 0.5592
event_avg_f           = 0.6709

--------------------------


python src/train.py   --npz-dir data/npz/guitarset_guitartechs_egdb_goat_idmt/split   --run-name guitarset_guitartechs_egdb_goat_idmt_onset_only   --test-num 6   --n-folds 7   --epoch 192 

python src/predict.py  guitarset_guitartechs_egdb_goat_idmt/guitarset_guitartechs_egdb_goat_idmt_onset 192 --test-num 06  --onset-threshold 0.80 --n-folds 7 --npz-dir data/npz/guitarset_guitartechs_egdb_goat_idmt/split

---------------------------

python src/train.py \
  --npz-dir data/npz/guitarset_guitartechs_egdb_reverb_aug_handpos_clean/split \
  --run-name guitarset_guitartechs_egdb_reverb_aug_handpos_clean \
  --target-balanced-sampling \
  --target-balanced-validation \
  --use-hand-position \
  --hand-position-fusion hidden+prior \
  --test-num 6 \
  --n-folds 7 \
  --epoch 192

python src/predict.py  guitarset_guitartechs_egdb_goat_idmt_reverb_aug_handpos_clean//guitarset_guitartechs_egdb_goat_idmt_reverb_aug_handpos_clean/ 192 --test-num 06  --onset-threshold 0.85 --n-folds 7 --npz-dir data/npz/guitarset_guitartechs_egdb_goat_idmt/split  --event-label-delay-ms 50   --event-label-window-ms 50   --event-string-window-ms 70
Headline metrics
frame_frame_avg_tab_f = 0.6722
frame_avg_onset_f     = 0.5917
event_avg_f           = 0.7286


--------------------------------------------- [important-unused]
CUDA_VISIBLE_DEVICES=1 python src/train.py   --npz-dir data/npz/guitarsetmicmix_egdbamps_guitartechs_goat_idmt_handpos_clean/split   --run-name guitarset_guitartechs_egdb_goat_idmt_onset_handpos_clean  --target-balanced-sampling  --target-balanced-sampling  --use-hand-position   --hand-position-fusion hidden+prior   --test-num 6   --n-folds 8   --epoch 192




-------------------------------------------

# Ablations

<!-- no hand-fused onset -->
CUDA_VISIBLE_DEVICES=0 python src/train.py   --npz-dir data/npz/guitarset_guitartechs_egdb_goat_idmt_reverb_aug_handpos_clean/split   --run-name guitarset_guitartechs_egdb_goat_idmt_reverb_aug_no_handfused_onset  --target-balanced-sampling   --target-balanced-validation   --use-hand-position   --hand-position-fusion prior   --test-num 6   --n-folds 7   --epoch 192

<!-- no hand at all-->
CUDA_VISIBLE_DEVICES=2 python src/train.py   --npz-dir data/npz/guitarset_guitartechs_egdb_goat_idmt_reverb_aug_handpos_clean/split   --run-name guitarset_guitartechs_egdb_goat_idmt_reverb_aug_no_hand   --target-balanced-sampling   --target-balanced-validation    --test-num 6   --n-folds 7   --epoch 192


python src/train.py   --npz-dir data/npz/guitarset_guitartechs_egdb_goat_idmt_reverb_aug_handpos_clean/split   --run-name guitarset_guitartechs_egdb_goat_idmt_reverb_aug_handpos   --target-balanced-sampling   --target-balanced-validation   --use-hand-position   --hand-position-fusion hidden+prior   --test-num 6   --n-folds 7   --epoch 192

<!-- no raw onset stream-->
CUDA_VISIBLE_DEVICES=3 python src/train.py   --npz-dir data/npz/guitarset_guitartechs_egdb_goat_idmt_reverb_aug_handpos_clean/split   --run-name guitarset_guitartechs_egdb_goat_idmt_reverb_aug_handpos_no_rawcqt_for_onset   --target-balanced-sampling   --target-balanced-validation   --use-hand-position   --hand-position-fusion hidden+prior   --test-num 6   --n-folds 7   --epoch 192 --no-onset-raw-features


--------------------------
# Guitarset


CUDA_VISIBLE_DEVICES=0 python src/train.py   --npz-dir data/npz/guitarset_handpos_clean/split   --run-name guitarset   --target-balanced-sampling   --target-balanced-validation   --use-hand-position   --hand-position-fusion prior   --n-folds 6   --epoch 192

CUDA_VISIBLE_DEVICES=2 python src/train.py   --npz-dir data/npz/guitarset/split   --run-name guitarset   --target-balanced-sampling   --target-balanced-validation   --use-hand-position   --hand-position-fusion prior   --n-folds 6   --epoch 192

CUDA_VISIBLE_DEVICES=2 python src/predict.py   guitarset/guitarset   192   --all-folds   --n-folds 6   --npz-dir data/npz/guitarset/split   --device cuda   --allow-missing-hand-pos   --event-decode-mode string_onset   --event-chord-group-ms 55   --event-label-delay-ms 20   --event-label-window-ms 90   --event-string-window-ms 50   --event-tab-threshold 0.30   --onset-threshold 0.80   --global-onset-threshold 0.80   --peak-pre-max-ms 30   --peak-post-max-ms 30   --peak-combine-ms 30   --use-global-onset-fallback   --global-fallback-max-notes 2   --repeat-same-fret-policy strong_onset   --min-repeat-ms 250   --repeat-onset-threshold 0.94   --repeat-global-threshold 0.75   --same-string-any-fret-min-ms 80   -v 


CUDA_VISIBLE_DEVICES=2 python src/train.py   --npz-dir data/npz/egdb/split   --run-name egdb   --target-balanced-sampling   --target-balanced-validatio
n   --use-hand-position   --hand-position-fusion prior --test-num 6   --n-folds 7   --epoch 192 --allow-missing-hand-pos 

CUDA_VISIBLE_DEVICES=2 python src/predict.py \
  egdb/egdb \
  192 \
  --all-folds \
  --n-folds 6 \
  --npz-dir data/npz/guitarset/split \
  --device cuda \
  --allow-missing-hand-pos \
  -v


CUDA_VISIBLE_DEVICES=2 python src/predict.py \
  guitarset/guitarset \
  192 \
  --all-folds \
  --n-folds 6 \
  --npz-dir data/npz/egdb/split \
  --device cuda \
  --allow-missing-hand-pos \
  -v

CUDA_VISIBLE_DEVICES=2 python src/predict.py \
  guitarset_handpos_clean/guitarset \
  192 
  [bad]

----------------------------------------------

 CUDA_VISIBLE_DEVICES=0 python src/train.py   --npz-dir data/npz/guitarset_handpos_clean/split   --run-name guitarset_handpos   --use-hand-position    --hand-position-fusion hidden+prior --n-folds 6 --epoch 192 

CUDA_VISIBLE_DEVICES=0 python src/predict.py  guitarset_handpos_clean/guitarset_handpos  192    --npz-dir data/npz/guitarset_handpos_clean/split --device cuda 


----------------------------------
<!-- IDMT no-hand test -->

CUDA_VISIBLE_DEVICES=0 python src/predict.py  guitarset_guitartechs_egdb_goat_idmt_reverb_aug_handpos_clean/guitarset_guitartechs_egdb_goat_idmt_reverb_aug_no_hand  192 --allow-missing-hand-pos --n-folds 7 --test-num 6  --npz-dir data/npz/idmt_06/split
[res]
notation_edit_avg_sim   = 0.4594
pitch_notation_edit_avg_sim = 0.7187
notation_lcs_avg_sim = 0.2811
pitch_notation_lcs_avg_sim = 0.5759
notation_overlap_avg_sim = 0.4710
pitch_notation_overlap_avg_sim = 0.7165










<!-- COMPACT -->

Compact audio-visual frame/onset model

audio waveform
      |
      v
CQT features
X : B x T x 192
      |
      +-------------------------+
      |                         |
      v                         v
raw onset projection       ConvStack + Conformer
192 -> 64                  192 -> 512
      |                         |
      v                         v
R : B x T' x 64            A : B x T' x 512
                                |
H : B x T' x 25                |
hand fret prior                 |
      |                         |
      v                         v
hand projection ----------> fusion
                          Z : B x T' x 512
                                |
      +-------------------------+-------------------------+
      |                                                   |
      v                                                   v
Frame-tab head                                      Onset heads
Linear + softmax                                    concat [Z ; R]
      |                                             4-layer gated TCN
      v                                                   |
P : B x T' x 6 x 21                       +---------------+---------------+
string/fret/rest probs                    |                               |
                                           v                               v
                                    global onset g                  per-string onset S
                                    B x T'                          B x T' x 6


-----------------------------
Audio-visual encoder + prediction heads

audio waveform
     |
     v
CQT: sr=22050, hop=512, bins=192
     |
     v
X : B x T x 192
     |
     +------------------------------------------------------+
     |                                                      |
     | raw CQT path for onset detection                     |
     |                                                      |
     |   X : B x T x 192                                    |
     |        |                                             |
     |        v                                             |
     |   Linear raw-feature projection                      |
     |   192 -> 64 + Dropout 0.10                           |
     |        |                                             |
     |        v                                             |
     |   R : B x T' x 64                                    |
     |   transient audio evidence                           |
     |                                                      |
     +--------------------------+---------------------------+
                                |
                                |
main encoded audio path         |
     |                          |
     v                          |
ConvStack                       |
Conv2D 1->32 -> Conv2D 32->32 -> MaxPool freq
Conv2D 32->64 -> MaxPool freq
     |
     v
B x 64 x T x 48
     |
     v
Flatten -> B x T x 3072
     |
     v
Linear 3072->512 + Dropout
     |
     v
1-layer Conformer encoder
1 head, self-attention over time
     |
     v
A : B x T' x 512
     |
     +-----------------------------+
                                   |
H : B x T' x 25                   |
hand-derived fret prior            |
     |                             |
     v                             v
Linear / LayerNorm / Dropout -> hidden-state fusion
                               A + projected H
                                   |
                                   v
Z : B x T' x 512
hand-conditioned encoded sequence
     |
     +--------------------------+--------------------------+
     |                                                     |
     |                                                     |
     v                                                     v

Frame-tab head                                      Onset input construction
--------------                                      ------------------------

Z : B x T' x 512                                    Z : B x T' x 512
     |                                               R : B x T' x 64
     v                                                    |
Linear frame-tab classifier                              |
512 -> 6*21                                              v
     |                                             concat [Z ; R]
     v                                             B x T' x 576
reshape + softmax
B x T' x 6 x 21
     |
     v
frame-tab probabilities
per frame, per string,
21 fret/rest classes


                                                    |
                                                    v
                                      +-------------+-------------+
                                      |                           |
                                      v                           v

                              Global onset head          Per-string onset head
                              ------------------         ---------------------

                              input: B x T' x 576        input: B x T' x 576
                                      |                           |
                                      v                           v
                              4 gated TCN blocks          4 gated TCN blocks
                              hidden dim = 64             hidden dim = 64
                              kernel size = 3             kernel size = 3
                              dilations = 1,2,4,8         dilations = 1,2,4,8
                                      |                           |
                                      v                           v
                              Linear 64 -> 1              Linear 64 -> 6
                                      |                           |
                                      v                           v
                              g : B x T'                  S : B x T' x 6

                              global note-start           string-specific
                              probability                 onset support
                              "is there an event          "which strings
                               at this frame?"             started here?"


Decoder distinction
-------------------

global onset g(t):
    Used for event timing.
    It answers:
        "Is there a note/chord onset at frame t?"

per-string onset S(t,s):
    Used as string support/gating.
    It answers:
        "Did string s have an onset near frame t?"

frame-tab P(t,s,f):
    Used for labels.
    It answers:
        "At frame t, what fret class is active on string s?"


Event decoder
-------------

1. peak-pick global onset g(t)
        |
        v
2. for each global onset frame t:
        |
        +--> use frame-tab P in a short post-onset window
        |    to choose the most likely fret for each string
        |
        +--> use per-string onset S around t
             to decide which strings are allowed to emit
        |
        v
3. emit selected string-fret notes
        |
        v
4. notes sharing the same global onset become a chord/event