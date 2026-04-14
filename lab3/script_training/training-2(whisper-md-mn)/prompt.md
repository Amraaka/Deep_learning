Fine Tune Whisper medium model on “Common Voice Scripted Speech 25.0 - Mongolian” dataset. Need to lower the WER score below 48%. 

Current setup: I have RTX 5080 16gb ram, downloaded the dataset on local for training from Mozilla dataset‘

Dataset Paths:
"
lab3/common_voice_mn
lab3/common_voice_mn/clips
lab3/common_voice_mn/clip_durations.tsv
lab3/common_voice_mn/dev.tsv
lab3/common_voice_mn/invalidated.tsv
lab3/common_voice_mn/other.tsv
lab3/common_voice_mn/README.md
lab3/common_voice_mn/reported.tsv
lab3/common_voice_mn/test.tsv
lab3/common_voice_mn/train.tsv
lab3/common_voice_mn/unvalidated_sentences.tsv
lab3/common_voice_mn/validated_sentences.tsv
lab3/common_voice_mn/validated.tsv"

Use only validated.tsv for training, validation, test.

I have an example notebook for fine tuning whisper on dataset. Can you write a code which is compatible with my dataset and setup. ask me options before starting to implement the code? I need sanity check before training. need to show tensorboard. huggingface token is in env. full training must fit into my GPU. create a new notebook in the same folder
