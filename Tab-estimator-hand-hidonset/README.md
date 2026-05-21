CUDA_VISIBLE_DEVICES=0 python src/train.py   --npz-dir data/npz/guitarset_guitartechs_egdb_goat_idmt_handpos_clean/split   --run-name hidonset_FULL   --use-hand-position   --hand-position-fusion hidden+prior   --test-num 6   --n-folds 7   --epoch 19


Input audio features
src_pad: (B, T_frame, F)
        |
        |  raw_features saved here
        |  raw_features = src_pad
        |
        v
+-------------------+
| Optional ConvStack |
+-------------------+
        |
        v
+----------------------+
| Shared Audio Encoder |
| Transformer/Conformer|
+----------------------+
        |
        | audio_memory: (B, T_enc, H)
        |
        +==============================================================+
        |                         |                                    |
        |                         |                                    |
        v                         v                                    v
 FRAME/TAB STREAM          HIDDEN ONSET STREAM                  NOTE/TAB STREAM
 hand-aware allowed        audio-only                            audio-only base
        |                         |                                    |
        |                         |                                    |
        |                  +------------------+                       |
        |                  | raw feature proj | <---- raw_features ----+
        |                  +------------------+
        |                         |
        |                         v
        |               resize raw proj to T_enc
        |                         |
        |                         v
        |          onset_input_mode:
        |            full    = concat(audio_memory, raw_proj)
        |            encoder = audio_memory only
        |            raw     = raw_proj only
        |                         |
        |                         v
        |                +----------------+
        |                | Onset TCN Head |
        |                | per-string     |
        |                +----------------+
        |                         |
        |                         +--> frame_onset_logits
        |                         |    (B, T_enc, 6)
        |                         |
        |                         +--> string_onset_features
        |                              (B, T_enc, onset_hidden_dim)
        |
        |                +----------------+
        |                | Onset TCN Head |
        |                | global         |
        |                +----------------+
        |                         |
        |                         +--> global_onset_logits
        |                         |    (B, T_enc)
        |                         |
        |                         +--> global_onset_features
        |                              (B, T_enc, onset_hidden_dim)
        |
        |                         |
        |                         v
        |          concat hidden onset features
        |          (string + global)
        |          (B, T_enc, 2 * onset_hidden_dim)
        |                         |
        |          optional detach:
        |          detach_onset_features_for_note
        |                         |
        |                         v
        |          BPM-aware decimation / pooling to note grid
        |          onset_note_features:
        |          (B, T_note, 2 * onset_hidden_dim)
        |                         |
        |                         |
        |                         v
        |                  hidden conditioning only
        |                  NO sigmoid threshold
        |                  NO event decoder
        |                         |
        +-------------------------+--------------------------+
                                  |
                                  v
audio_memory ----------------> BPM-aware note decimation
(B, T_enc, H)                  or interpolation
                                  |
                                  v
                         decimated_memory
                         (B, T_note, H)
                                  |
                                  v
                   +-------------------------------+
                   | NoteOnsetConditioning          |
                   | gated add:                     |
                   | note_hidden + gate * onset_proj|
                   +-------------------------------+
                                  |
                                  v
                   optional note hidden hand fusion
                   only if note_hidden_hand_fusion=True
                   default: OFF
                                  |
                                  v
                         +--------------+
                         | note_encoder |
                         | Conformer    |
                         +--------------+
                                  |
                                  v
                     +-----------------------+
                     | note_tab_output_layer |
                     +-----------------------+
                                  |
                                  v
                         softmax by string
                                  |
                                  v
                    optional hand-position prior
                    REST-PRESERVING by default
                                  |
                                  v
                         note_tab_pred
                         (B, T_note, 6, 21)
                         MAIN FINAL OUTPUT

                         Frame/tab side in more detail:

audio_memory
(B, T_enc, H)
     |
     | optional frame hidden hand fusion
     | uses frame_hand_pos
     v
frame_memory
     |
     v
frame_tab_output_layer
     |
     v
reshape to (B, T_enc, 6, 21)
     |
     v
softmax by string
     |
     v
optional normal hand-position prior
     |
     v
frame_tab_pred
(B, T_enc, 6, 21)

Onset branch in more detail:

audio_memory                         raw_features
(B, T_enc, H)                        (B, T_frame, F)
     |                                    |
     |                                    v
     |                            raw feature projection
     |                                    |
     |                                    v
     |                            raw_proj resized to T_enc
     |                                    |
     +-------------------+----------------+
                         |
                         v
                  onset_input_mode

        full:    concat(audio_memory, raw_proj)
        encoder: audio_memory
        raw:     raw_proj

                         |
             +-----------+------------+
             |                        |
             v                        v
   per-string OnsetTCNHead     global OnsetTCNHead
             |                        |
             v                        v
 frame_onset_logits        global_onset_logits
 diagnostic / aux loss     diagnostic / aux loss
 not final decoder         not final decoder

             |                        |
             v                        v
 string_onset_features     global_onset_features
             |                        |
             +-----------+------------+
                         |
                         v
              concat hidden features
                         |
                         v
              decimate to note grid
                         |
                         v
              condition note stream