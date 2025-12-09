# src/datasets/custom_dir_dataset.py

from pathlib import Path
from typing import List, Dict, Any

from torch.utils.data import Dataset


class CustomDirDataset(Dataset):
    
    def __init__(
        self,
        root_dir: str,
        transcription_subdir: str = "transcriptions",
        encoding: str = "utf-8",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transcriptions_dir = self.root_dir / transcription_subdir
        self.encoding = encoding

        if not self.transcriptions_dir.is_dir():
            raise ValueError(
                f"`{self.transcriptions_dir}` не существует или не является директорией"
            )

        self._files: List[Path] = sorted(
            p for p in self.transcriptions_dir.glob("*.txt") if p.is_file()
        )

        if not self._files:
            raise ValueError(
                f"В `{self.transcriptions_dir}` не найдено ни одного .txt файла"
            )

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self._files[idx]
        utt_id = path.stem

        with path.open("r", encoding=self.encoding) as f:
            text = f.read().strip()

        return {
            "utt_id": utt_id,
            "text": text,
        }
