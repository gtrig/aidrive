"""
Track registry — discovers and loads track metadata from self-contained track
folders under assets/tracks/.

Folder layout expected for every track:
    assets/tracks/<name>/
        background.png    # grass / outer image
        tarmac.png        # road surface image
        track.npy         # collision line segments (N, 2, 2) int32
        gates.npy         # gate checkpoints (G, 2, 2) int32
        meta.json         # see TrackMeta below

meta.json schema:
    {
        "start": {"x": 265, "y": 130, "heading": 90},
        "image_scale": 1.3
    }

Adding a new track requires no code changes; just drop the folder here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

TRACKS_DIR: Path = Path(__file__).resolve().parent.parent / 'assets' / 'tracks'


@dataclass
class TrackMeta:
    name: str
    folder: Path
    start_x: float
    start_y: float
    start_heading: float
    image_scale: float = 1.3

    @property
    def track_npy(self) -> Path:
        return self.folder / 'track.npy'

    @property
    def gates_npy(self) -> Path:
        return self.folder / 'gates.npy'

    @property
    def background_png(self) -> Path:
        return self.folder / 'background.png'

    @property
    def tarmac_png(self) -> Path:
        return self.folder / 'tarmac.png'

    @property
    def meta_json(self) -> Path:
        return self.folder / 'meta.json'


def list_tracks() -> list[str]:
    """Return track names (sorted) of all valid track folders discovered."""
    if not TRACKS_DIR.exists():
        return []
    return sorted(
        d.name
        for d in TRACKS_DIR.iterdir()
        if d.is_dir() and (d / 'track.npy').exists() and (d / 'meta.json').exists()
    )


def load(name: str) -> TrackMeta:
    """Load and return TrackMeta for the named track.

    Raises FileNotFoundError if the folder or meta.json does not exist.
    Raises ValueError if meta.json is missing required fields.
    """
    folder = TRACKS_DIR / name
    meta_path = folder / 'meta.json'
    if not meta_path.exists():
        raise FileNotFoundError(
            f'Track "{name}" not found at {folder}. '
            f'Available tracks: {list_tracks()}'
        )
    with meta_path.open() as fh:
        data = json.load(fh)

    try:
        start = data['start']
        return TrackMeta(
            name=name,
            folder=folder,
            start_x=float(start['x']),
            start_y=float(start['y']),
            start_heading=float(start['heading']),
            image_scale=float(data.get('image_scale', 1.3)),
        )
    except KeyError as exc:
        raise ValueError(
            f'meta.json for track "{name}" is missing field: {exc}'
        ) from exc


def write_meta(folder: Path, start_x: float, start_y: float,
               start_heading: float, image_scale: float = 1.3) -> None:
    """Write (or overwrite) meta.json in the given track folder."""
    folder.mkdir(parents=True, exist_ok=True)
    meta = {
        'start': {
            'x': round(float(start_x), 3),
            'y': round(float(start_y), 3),
            'heading': round(float(start_heading), 3),
        },
        'image_scale': image_scale,
    }
    with (folder / 'meta.json').open('w') as fh:
        json.dump(meta, fh, indent=2)
