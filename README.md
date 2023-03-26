# speaker-diarization

This repository contains the Cog definition files for the associated speaker diarization model [deployed on Replicate](https://replicate.com/meronym/speaker-diarization).

This model receives an audio file and identifies the individual speakers within the recording. The output is a list of annotated speech segments, along with global information about the number of detected speakers and an embedding vector for each speaker to describe the quality of his/her voice.

## Model description

The model is based on a pre-trained speaker diarization pipeline from the [`pyannote.audio`](pyannote.github.io) package, with a post-processing layer that cleans up the output segments and computes input-wide speaker embeddings.

`pyannote.audio` is an open-source toolkit written in Python for speaker diarization. Based on the [PyTorch](pytorch.org) machine learning framework, it provides a set of trainable end-to-end neural building blocks that can be combined and jointly optimized to build speaker diarization pipelines.

The main pipeline makes use of:

- `pyannote/segmentation` for permutation-invariant speaker classification on temporal segments
- `speechbrain/spkrec-ecapa-voxceleb` for generating speaker embeddings
- `AgglomerativeClustering` for determining individual speakers

See [this post](https://herve.niderb.fr/fastpages/2022/10/23/One-speaker-segmentation-model-to-rule-them-all.html) (from `pyannote.audio`'s author) for more details.

## Output format

The model outputs a single `output.json` file with the following structure:

```json
{
  "segments": [
    {
      "speaker": "A",
      "start": "0:00:00.497812",
      "stop": "0:00:49.452188"
    },
    {
      "speaker": "B",
      "start": "0:00:49.857188",
      "stop": "0:01:30.981562"
    }
  ],
  "speakers": {
    "count": 2,
    "labels": [
      "A",
      "B"
    ],
    "embeddings": {
      "A": [<array of 192 floats>],
      "B": [<array of 192 floats>]
    }
  }
}
```

## Performance

The current T4 deployment has an average processing time of 12x (relative to the length of the audio input) - e.g. it will take the model approx. 1 minute of computation to process 12 minutes of audio.

## Intended use

Data augmentation and segmentation for a variety of transcription and captioning tasks (e.g. interviews, podcasts, meeting recordings, etc.). Speaker recognition can be implemented by matching the speaker embeddings against a database of known speakers.

## Ethical considerations

This model may have biases based on the data it has been trained on. It is important to use the model in a responsible manner and adhere to ethical and legal standards.

## Caveats and recommendations

This model has only been tested with `wav` and `flac` input files.

## Citations

If you use `pyannote.audio` please use the following citations:

```bibtex
@inproceedings{Bredin2020,
  Title = {{pyannote.audio: neural building blocks for speaker diarization}},
  Author = {{Bredin}, Herv{\'e} and {Yin}, Ruiqing and {Coria}, Juan Manuel and {Gelly}, Gregory and {Korshunov}, Pavel and {Lavechin}, Marvin and {Fustes}, Diego and {Titeux}, Hadrien and {Bouaziz}, Wassim and {Gill}, Marie-Philippe},
  Booktitle = {ICASSP 2020, IEEE International Conference on Acoustics, Speech, and Signal Processing},
  Year = {2020},
}
```

```bibtex
@inproceedings{Bredin2021,
  Title = {{End-to-end speaker segmentation for overlap-aware resegmentation}},
  Author = {{Bredin}, Herv{\'e} and {Laurent}, Antoine},
  Booktitle = {Proc. Interspeech 2021},
  Year = {2021},
}
```
