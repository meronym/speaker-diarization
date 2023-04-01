"""
download model weights to /data
wget wget -O - https://pyannote-speaker-diarization.s3.eu-west-2.amazonaws.com/data-2023-03-25-02.tar.gz | tar xz -C /
"""
import json
import tempfile

from cog import BasePredictor, Input, Path
from pyannote.audio.pipelines import SpeakerDiarization

from lib.diarization import DiarizationPostProcessor
from lib.audio import AudioPreProcessor


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.diarization = SpeakerDiarization(
            segmentation="/data/pyannote/segmentation/pytorch_model.bin",
            embedding="/data/speechbrain/spkrec-ecapa-voxceleb",
            clustering="AgglomerativeClustering",
            segmentation_batch_size=32,
            embedding_batch_size=32,
            embedding_exclude_overlap=True,
        )
        self.diarization.instantiate({
            "clustering": {
                "method": "centroid",
                "min_cluster_size": 15,
                "threshold": 0.7153814381597874,
            },
            "segmentation": {
                "min_duration_off": 0.5817029604921046,
                "threshold": 0.4442333667381752,
            },
        })
        self.diarization_post = DiarizationPostProcessor()
        self.audio_pre = AudioPreProcessor()

    def run_diarization(self):
        closure = {'embeddings': None}

        def hook(name, *args, **kwargs):
            if name == "embeddings" and len(args) > 0:
                closure['embeddings'] = args[0]

        print('diarizing audio file...')
        diarization = self.diarization(self.audio_pre.output_path, hook=hook)
        embeddings = {
            'data': closure['embeddings'],
            'chunk_duration': self.diarization.segmentation_duration,
            'chunk_offset': self.diarization.segmentation_step * self.diarization.segmentation_duration,
        }
        return self.diarization_post.process(diarization, embeddings)

    def predict(
        self,
        audio: Path = Input(description="Audio file",
                            default="https://pyannote-speaker-diarization.s3.eu-west-2.amazonaws.com/lex-levin-4min.mp3"),
    ) -> Path:
        """Run a single prediction on the model"""

        self.audio_pre.process(audio)

        if self.audio_pre.error:
            print(self.audio_pre.error)
            result = self.diarization_post.empty_result()
        else:
            result = self.run_diarization()

        self.audio_pre.cleanup()

        output = Path(tempfile.mkdtemp()) / "output.json"
        with open(output, "w") as f:
            f.write(json.dumps(result, indent=2))
        return output
