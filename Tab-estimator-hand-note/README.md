python src/train.py \
  --npz-dir data/npz/guitarset_guitartechs_egdb_goat_idmt_handpos_clean/split \
  --run-name event_assembly_FULL \
  --use-hand-position \
  --hand-position-fusion hidden+prior \
  --test-num 6 \
  --n-folds 7 \
  --epoch 192

python src/predict.py \
  guitarset_guitartechs_egdb_goat_idmt_handpos_clean/event_assembly_FULL \
  192 \
  --test-num 06 \
  --n-folds 7 \
  --npz-dir data/npz/guitarset_guitartechs_egdb_goat_idmt_handpos_clean/split  


audio features
     |
     v
shared audio encoder
     |
     +----------------------------+
     |                            |
     v                            v
frame-tab stream              onset streams
                              per-string + global TCN
                                      |
                                      v
                              onset hidden features
                                      |
frame-tab hidden + onset hidden + encoder memory
                                      |
                                      v
                chord-aware event assembly head
                non-causal gated TCN, default 6 layers
                                      |
        +-------------+--------------+-------------+
        |             |              |             |
        v             v              v             v
 event logits   string mask    fret logits   event type
                               per string    none/single/chord/arpeggio