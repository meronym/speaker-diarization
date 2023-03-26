"""
download model weights to /data
wget wget -O - https://pyannote-speaker-diarization.s3.eu-west-2.amazonaws.com/data-2023-03-25-02.tar.gz | tar xz -C /
"""
import json
import tempfile

from cog import BasePredictor, Input, Path
from pyannote.audio.pipelines import SpeakerDiarization

from lib.utils import DiarizationPostProcessor


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

    def predict(
        self,
        audio: Path = Input(description="Audio file"),
    ) -> Path:
        """Run a single prediction on the model"""
        closure = {'embeddings': None}

        def hook(name, *args, **kwargs):
            if name == "embeddings" and len(args) > 0:
                closure['embeddings'] = args[0]

        diarization = self.diarization(audio, hook=hook)
        embeddings = {
            'data': closure['embeddings'],
            'chunk_duration': self.diarization.segmentation_duration,
            'chunk_offset': self.diarization.segmentation_step * self.diarization.segmentation_duration,
        }
        result = self.diarization_post.process(diarization, embeddings)

        output = Path(tempfile.mkdtemp()) / "output.json"
        with open(output, "w") as f:
            f.write(json.dumps(result, indent=2))
        return output
