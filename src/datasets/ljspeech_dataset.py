import json
import os
import shutil
import hashlib
from pathlib import Path

import torchaudio
import wget
from tqdm import tqdm
from typing import Literal

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH


class LJSpeech(BaseDataset):
    def __init__(self, data_dir=None, partition=Literal["train", "val"], *args, **kwargs):
        if data_dir is None:
            data_dir = ROOT_PATH / "data"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = Path(data_dir)
        index = self._get_or_load_index(partition)

        super().__init__(index, *args, **kwargs)

    def _load(self):
        arch_path = self._data_dir / "LJSpeech-1.1.tar.bz2"
        print("Loading LJSpeech-1.1 dataset")
        wget.download(
            "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
            str(arch_path),
        )
        shutil.unpack_archive(arch_path, self._data_dir)

        extracted_dir = self._data_dir / "LJSpeech-1.1"

        for fpath in extracted_dir.iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))

        os.remove(str(arch_path))
        shutil.rmtree(str(extracted_dir))

    def _get_or_load_index(self, partition: Literal['train', 'val']):
        index_path = self._data_dir / "index_ljspeech.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index()
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)

        def text_hash(item: dict) -> int:
            h = hashlib.md5(item["text"].encode("utf-8")).hexdigest()
            return int(h, 16)

        index_sorted = sorted(index, key=text_hash)

        if partition == "val":
            return index_sorted[:200]

        return index_sorted[200:]

    def _create_index(self):
        metadata_path = self._data_dir / "metadata.csv"
        wavs_dir = self._data_dir / "wavs"

        if not metadata_path.exists() or not wavs_dir.exists():
            self._load()

        assert metadata_path.exists(), f"metadata.csv not found in {self._data_dir}"
        assert wavs_dir.exists(), f"wavs/ dir not found in {self._data_dir}"

        index = []
        with metadata_path.open(encoding="utf-8") as f:
            for line in tqdm(f, desc="Building LJSpeech index"):
                line = line.strip()
                if not line:
                    continue

                parts = line.split("|")
                if len(parts) < 3:
                    continue
                utt_id, raw_text, norm_text = parts[0], parts[1], parts[2]

                wav_path = wavs_dir / f"{utt_id}.wav"
                if not wav_path.exists():
                    tqdm.write(f'Wav path not found {wav_path}')
                    continue

                info = torchaudio.info(str(wav_path))
                duration = info.num_frames / info.sample_rate

                index.append(
                    {
                        "path": str(wav_path),
                        "text": norm_text,
                        "duration": duration,
                    }
                )

        return index
