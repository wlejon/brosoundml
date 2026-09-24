# Laya audio adapter: training data and model licences

The adapter (an MLP projector from Qwen3-ASR encoder latents into Laya's
input embedding space) is to be released under **Apache-2.0**, together with a
NOTICE file crediting the CC BY sources below. Every byte used for training
must allow training on it and releasing the resulting model commercially. Rule
applied: public domain, CC0, CC BY (any version) and Apache-2.0/MIT sources
are admitted; anything non-commercial (NC), share-alike (SA), no-derivatives
(ND), research-only or without a stated licence is excluded. Every licence
below was checked at its source (dataset page, the licence file inside the
archive, or the model card) on 2026-09-23.

The trained adapter contains no audio and no text from these datasets, only
weights learned from them. CC BY 4.0 treats the weights as adapted material
at most, which needs attribution (the NOTICE) but no licence of its own.

## Models the adapter depends on

| component | role | licence | source |
|---|---|---|---|
| Qwen3-ASR-0.6B (AuT encoder) | frozen audio encoder whose latents the projector reads | Apache-2.0 | huggingface.co/Qwen/Qwen3-ASR-0.6B (model card) |
| Laya (English, ModernBERT-large) | frozen decision model | Apache-2.0 | D:/projects/laya README / model card; ModernBERT-large is Apache-2.0 |
| Laya multilingual (mmBERT-base) | frozen decision model | Apache-2.0 | D:/projects/laya/multilingual; mmBERT-base is MIT |
| NVIDIA parakeet-tdt-0.6b-v3 | produced the word timings (labels) for LibriTTS-R and VoxPopuli | CC BY 4.0 | huggingface.co/nvidia/parakeet-tdt-0.6b-v3 (model card) |

Parakeet's outputs (word start/end times) are training labels, not shipped
data. CC BY 4.0 permits commercial use of outputs, and it is credited in the
NOTICE all the same.

## Speech data used

Sizes are what is stored under D:/datasets, compressed. LibriTTS-R already
existed there and does not count against the 10 GB budget. New audio in
total: **5.3 GB** (VoxPopuli 2.0, AMI 1.3, MSWC 1.1, OpenSLR 28 0.5, MUSAN
0.47), all Ogg Opus or FLAC. Archives were streamed and cut on the fly,
never stored whole.

| source | licence | URL | subset taken | stored | why it is compatible |
|---|---|---|---|---|---|
| LibriTTS-R | CC BY 4.0 | openslr.org/141 | train-clean-100 (training), dev-clean (eval) | pre-existing | CC BY: attribution only |
| VoxPopuli, transcribed ASR subset (en, de, es, fr) | transcriptions CC0; audio (c) European Union, reuse authorised with the source acknowledged | github.com/facebookresearch/voxpopuli; europarl.europa.eu legal notice | en: 2018 sessions, a random train sample capped at 50 h, plus all dev/test; de/es/fr: 2017-2018 sessions, train capped at 40 h, plus all dev/test. The "invalid" split is never used. Segments overlapping or abutting their predecessor are dropped by the cutter, so the kept train hours are en 37.8, de 29.0, es 23.7, fr 24.8 (dev/test 0.7-1.9 h each) | 2.0 GB (en 0.65, de 0.51, es 0.41, fr 0.44) | CC0 text; the EP legal notice authorises reproduction including commercial reuse, provided the source is acknowledged (done in the NOTICE) |
| AMI Meeting Corpus, SDM (Array1-01) | CC BY 4.0 | groups.inf.ed.ac.uk/ami/corpus (LICENCE.txt in the annotation release) | all 169 meetings with SDM audio; word annotations (words/*.words.xml). Kaldi full-corpus split: dev and eval meetings are never trained on (train 73.1 h, dev 8.6 h, test 8.2 h) | 1.3 GB | CC BY: attribution only |
| Multilingual Spoken Words (MSWC) | CC BY 4.0 (derived from Common Voice, CC0) | mlcommons.org/datasets/multilingual-spoken-words; storage.googleapis.com/public-datasets-mswc | en: 1,500 random words (150 per bucket of 1..256 training clips, plus 150 never-trained words), 8 test clips each from held-out speakers (76,650 train / 12,000 test clips); de/es/fr: 350 random words each (100 per bucket of 4/16/64 training clips, plus 50 never-trained words), 8 test clips each (8,400 / 2,800 clips per language). Test speakers are 1/8 of all speakers, by id, and never contribute a training clip | 1.1 GB (en 0.80, de/es/fr 0.10 each) | CC BY: attribution only |
| Timers and Such v1.0 (real recordings only) | CC0 1.0 (the LICENSE file inside the v1.0 archive; the older v0.1 Zenodo record says CC BY 4.0, which would also be admissible) | zenodo.org/records/4623772 (L. Lugosch et al., SpeechBrain) | train-real (training), dev-real and test-real (eval); the synthetic TTS subsets are not downloaded. Train 1,640 commands, test 240 (disjoint speakers per the dataset split) | 0.29 GB | CC0: no conditions (credited in the NOTICE anyway) |

The free-form distillation round (docs/laya-audio.md, "Free-form questions")
trains on the same audio as above plus Timers and Such. Its targets come from
text Laya (English, Apache-2.0) answering bank questions about the gold
transcript of each window: model outputs of an Apache-2.0 model, used as
labels and not shipped. No label from any evaluation-only set below is ever
used in training.

## Non-speech data used (augmentation and negatives)

| source | licence | URL | subset taken | stored | why it is compatible |
|---|---|---|---|---|---|
| OpenSLR 28, simulated_rirs | Apache-2.0 (SLR 26, image-method simulation) | openslr.org/28, openslr.org/26 | the first 10 RIRs of each of the 600 simulated rooms (small/medium/large): 6,000 | 0.50 GB (with the noises) | Apache-2.0 |
| OpenSLR 28, pointsource_noises | public domain (its LICENSE: "all selected recordings were marked as in the Public Domain on Free Sound") | openslr.org/28 | all 843 | (with RIRs) | public domain |
| MUSAN noise/sound-bible | per file: CC BY 3.0 (66) or public domain (20) | openslr.org/17 (MUSAN LICENSE files) | 86 of 87 files; noise-sound-bible-0000 has no licence entry and was deleted | 0.47 GB (with the music) | CC BY / PD, credited per file in ATTRIBUTIONS.tsv |
| MUSAN music | per file, from MUSAN's LICENSE blocks | openslr.org/17 | only PD / CC0 / CC BY tracks (327 of the 645 annotated), and of those only the 261 without vocals are used | (with MUSAN) | CC BY / PD, credited per file in ATTRIBUTIONS.tsv |

15 % of the non-speech files (by a hash of the file name) are held out for
the non-speech negative test sets; the rest are both negatives in training
and the noise/music pool for augmentation.

## Evaluation-only data (NEVER trained on)

These sets measure the free-form questions against real human labels. The
SLURP and CREMA-D licences do not allow a commercial model to be trained on
them, so they are used only to score a finished projector. No training script reads them
(tools/laya_audio_fetch/train_*.sh take none of their caches), and nothing
derived from them ships. Checked 2026-09-24.

| source | licence | URL | subset taken | stored | status |
|---|---|---|---|---|---|
| SLURP | text and annotations CC BY 4.0; **audio CC BY-NC 4.0** (the repository's LICENSE.txt) | github.com/pswietojanski/slurp; zenodo.org/record/4274930 | test split, slurp_real only, the first recording of each test utterance: 2,780 recordings (the archive stream ended early; the possibly truncated last file was deleted) | 0.15 GB | **eval-only** (non-commercial audio) |
| CREMA-D | Open Database License + Database Contents License (share-alike database licence) | github.com/CheyneyComputerScience/CREMA-D | acted emotional sentences (6 emotions, intensity labels): all 7,442 clips via Git LFS; the eval samples windows from them | 0.59 GB | **eval-only** (share-alike) |
| AMI manual annotations: dialogue acts, named entities, topic segmentation, ontologies | CC BY 4.0 (same release as the AMI words) | groups.inf.ed.ac.uk/ami/corpus | the manual-annotation release, of which only the labels of the AMI test meetings are read (12 of them carry dialogue acts; EN2002 has none) | 0.05 GB | eval labels only. CC BY would allow training, but they were deliberately kept for evaluation |

Emotion / tone-of-voice sets were searched for a licence-clean training set
and none was found: RAVDESS (CC BY-NC-SA 4.0), TESS (CC BY-NC-ND 4.0),
CREMA-D (ODbL, share-alike), ESD, IEMOCAP and MSP-Podcast (research-only
licence agreements). The tone-of-voice questions are therefore measured
zero-shot only, on CREMA-D, and never trained.

New downloads for this round: 1.0 GB (SLURP 0.15, Timers 0.29, CREMA-D 0.59,
AMI annotations 0.05).

## Considered and dropped

| source | why dropped |
|---|---|
| People's Speech (MLCommons) | The original tar archives are no longer distributed; Hugging Face serves parquet only, and reading it needs Python or a parquet library (the brief allows dropping a source that realistically needs Python). The licence is per configuration (`clean` CC BY 4.0, `clean_sa` CC BY-SA 4.0) rather than per row, and the validation/test splits are shared between the two, so their licence is ambiguous. |
| OpenSLR 28, real_rirs_isotropic_noises | Although SLR 28 is labelled Apache-2.0, this folder bundles the RWCP sound scene database (SLR 13: "research and development use only"), the Aachen AIR database (no licence stated) and REVERB challenge RIRs. Not clean for a commercial release. |
| MUSAN music under CC BY-SA / BY-NC / BY-ND or unlabelled | share-alike, non-commercial, no-derivatives or unknown |
| MUSAN noise/free-sound | not needed: the same recordings ship, public domain, as OpenSLR 28 pointsource_noises |
| MUSAN speech | Not needed (LibriVox/US government speech would duplicate the speech domains) |
| noise-sound-bible-0000 | no entry in MUSAN's sound-bible LICENSE |
| SLURP audio, CREMA-D, RAVDESS, TESS, ESD, IEMOCAP, MSP-Podcast as training data | non-commercial, share-alike, no-derivatives or research-only (SLURP and CREMA-D are used for evaluation only, see above) |
| Timers and Such synthetic subsets | not needed (TTS speech); the 13 GB archive is range-read for the real subsets only |

## Files

- `ATTRIBUTIONS.tsv`: title, author, licence and URL for every MUSAN file
  trained on (tools/laya_audio_fetch/attributions.sh).
- `NOTICE`: the release NOTICE draft.
- Reproduction: tools/laya_audio_fetch/*.sh in brosoundml fetch each source
  and apply the licence filters; build_mix.sh builds the alignments and caches.
  fetch_timers.sh, fetch_slurp.sh (eval-only) and fetch_cremad.sh (eval-only)
  fetch the free-form round's sets; build_questions.sh builds their labels,
  alignments and caches.
