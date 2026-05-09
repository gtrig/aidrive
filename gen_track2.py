"""
Track 2 generator.

Builds a self-contained track folder under assets/tracks/<name>/:
  background.png    (outer green area, RGBA 1000x578)
  tarmac.png        (road surface, RGBA 1000x578)
  track.npy         (line segments for collision / Shapely)
  gates.npy         (54 gate segments perpendicular to the circuit)
  meta.json         (start position + image scale)

The circuit is defined as a smooth closed centreline.  Shapely buffers it by
±half_width to derive outer/inner rings.  Images are drawn with PIL.

Run:
    .venv/bin/python gen_track2.py
    .venv/bin/python gen_track2.py --name my_oval
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from shapely.geometry import LinearRing, LineString, MultiPolygon, Polygon
from shapely.ops import unary_union

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

IMG_W, IMG_H   = 1000, 578      # image pixel size (matches track1)
SCALE          = 1.3            # sprite scale applied in track.py
COORD_W        = IMG_W * SCALE  # ~1300 — coordinate space width
COORD_H        = IMG_H * SCALE  # ~751  — coordinate space height

TRACK_WIDTH    = 75             # road width in coordinate units
N_GATES        = 54             # number of gate checkpoints

GRASS_COLOR    = (72, 130, 40, 255)    # dark green
ROAD_COLOR     = (60, 60, 60, 255)     # dark asphalt
KERB_COLOR     = (200, 200, 200, 255)  # light grey kerb strip
TRANSPARENT    = (0, 0, 0, 0)
LINE_COLOR     = (255, 255, 255, 200)  # white centre line (dashed)

# ------------------------------------------------------------------
# Centreline definition (coordinate space: x right, y up like Pyglet)
# ------------------------------------------------------------------

def _arc(cx, cy, r, a0_deg, a1_deg, n=40):
    """Return n points on an arc from angle a0 to a1 (degrees, CCW)."""
    a0, a1 = math.radians(a0_deg), math.radians(a1_deg)
    # Use linspace but handle wrap-around correctly
    delta = a1 - a0
    if abs(delta) < 1e-6:
        return []
    angles = np.linspace(a0, a1, n, endpoint=False)
    return [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in angles]


def build_centreline() -> list[tuple[float, float]]:
    """
    Track 2 circuit layout (viewed from above, Pyglet Y-up coordinates):

    ──────────── back straight ────────────
    ↑ left                        right ↓
    hairpin                      hairpin
    ↓                                  ↑
    ──── chicane ────── front straight ────

    Counter-clockwise when drawn on screen (Y up).
    """
    pts: list[tuple[float, float]] = []

    cx = COORD_W / 2   # ~650
    cy = COORD_H / 2   # ~375

    # ---- Front straight (bottom): left → right ----
    front_y = 130
    pts += [(x, front_y) for x in np.linspace(280, 980, 30)]

    # ---- Right hairpin: wide 180° turn (going from bottom → top) ----
    rh_cx = 980
    rh_cy = (front_y + 620) / 2   # midpoint between front and back Y
    rh_r  = (620 - front_y) / 2
    pts += _arc(rh_cx, rh_cy, rh_r, -90, 90, 40)   # -90° (bottom) → +90° (top)

    # ---- Back straight with chicane (top): right → left ----
    back_y = 620
    # gentle S-chicane in the middle of the back straight
    chicane_pts = [
        (rh_cx, back_y),
        (860,   back_y),
        (790,   back_y + 55),   # jink right
        (710,   back_y + 55),
        (640,   back_y - 10),
        (560,   back_y - 10),
        (490,   back_y + 55),   # jink left
        (410,   back_y + 55),
        (340,   back_y),
        (280,   back_y),
    ]
    pts += chicane_pts

    # ---- Left hairpin: tighter 180° turn (top → bottom) ----
    lh_cx = 280
    lh_cy = rh_cy
    lh_r  = rh_r
    pts += _arc(lh_cx, lh_cy, lh_r, 90, 270, 40)   # +90° (top) → +270° (bottom)

    return pts


def smooth_polygon(pts: list, iterations: int = 3) -> list:
    """Chaikin corner-cutting smoothing."""
    arr = np.array(pts, dtype=float)
    for _ in range(iterations):
        q = 0.75 * arr + 0.25 * np.roll(arr, -1, axis=0)
        r = 0.25 * arr + 0.75 * np.roll(arr, -1, axis=0)
        arr = np.empty((len(arr) * 2, 2))
        arr[0::2] = q
        arr[1::2] = r
    return list(map(tuple, arr))


def resample(pts: list, spacing: float = 15.0) -> list:
    """Resample a closed polygon to approximately `spacing` units between points."""
    arr  = np.array(pts, dtype=float)
    # close the loop
    arr  = np.vstack([arr, arr[0]])
    diffs = np.diff(arr, axis=0)
    dists = np.hypot(diffs[:, 0], diffs[:, 1])
    cum   = np.concatenate([[0], np.cumsum(dists)])
    total = cum[-1]
    target = np.arange(0, total, spacing)
    xs = np.interp(target, cum, arr[:, 0])
    ys = np.interp(target, cum, arr[:, 1])
    return list(zip(xs, ys))


# ------------------------------------------------------------------
# Build track geometry
# ------------------------------------------------------------------

GRASS_MARGIN = 20   # px by which the grass overflows beyond each road edge

def build_rings(centre: list, half_width: float):
    """
    Return (outer_pts, inner_pts, grass_outer_pts, grass_inner_pts).

    outer_pts / inner_pts        – road boundaries used for .npy and tarmac image
    grass_outer_pts              – outer boundary expanded by GRASS_MARGIN
    grass_inner_pts              – inner boundary shrunk by GRASS_MARGIN
                                   (makes green visible on both sides of the road)
    """
    poly  = Polygon(centre)
    outer = poly.buffer(half_width,               join_style=1, cap_style=1)
    inner = poly.buffer(-half_width,              join_style=1, cap_style=1)
    grass_outer = poly.buffer(half_width + GRASS_MARGIN, join_style=1, cap_style=1)
    grass_inner = poly.buffer(-(half_width - GRASS_MARGIN), join_style=1, cap_style=1)

    def _coords(geom):
        if isinstance(geom, (MultiPolygon,)):
            geom = max(geom.geoms, key=lambda g: g.area)
        return list(geom.exterior.coords)

    return _coords(outer), _coords(inner), _coords(grass_outer), _coords(grass_inner)


def polygon_to_segments(pts: list, spacing: float = 15.0) -> np.ndarray:
    """Convert a ring (list of (x,y)) to an array of shape (N,2,2) int32."""
    resampled = resample(pts, spacing)
    n = len(resampled)
    segs = np.zeros((n, 2, 2), dtype=np.int32)
    for i in range(n):
        p0 = resampled[i]
        p1 = resampled[(i + 1) % n]
        segs[i, 0] = [round(p0[0]), round(p0[1])]
        segs[i, 1] = [round(p1[0]), round(p1[1])]
    return segs


# ------------------------------------------------------------------
# Gate generation
# ------------------------------------------------------------------

def build_gates(centre_pts: list, n_gates: int, track_width: float) -> np.ndarray:
    """
    Place n_gates evenly along the centreline.  Each gate is a line segment
    perpendicular to the track direction, of length ~track_width.
    Returns shape (n_gates, 2, 2) int32.
    """
    resampled = resample(centre_pts, spacing=1.0)   # very dense for accuracy
    n = len(resampled)
    arr = np.array(resampled, dtype=float)
    indices = np.linspace(0, n - 1, n_gates, endpoint=False, dtype=int)

    gates = np.zeros((n_gates, 2, 2), dtype=np.int32)
    half = track_width * 0.55   # slightly wider than road half-width

    for g, idx in enumerate(indices):
        cx, cy = arr[idx]
        nxt     = arr[(idx + 1) % n]
        prv     = arr[(idx - 1) % n]
        # tangent direction
        tx = nxt[0] - prv[0]
        ty = nxt[1] - prv[1]
        length = math.hypot(tx, ty) or 1.0
        # normal (perpendicular to tangent)
        nx = -ty / length
        ny =  tx / length
        p1 = (round(cx + nx * half), round(cy + ny * half))
        p2 = (round(cx - nx * half), round(cy - ny * half))
        gates[g] = [p1, p2]

    return gates


# ------------------------------------------------------------------
# Image generation (coordinate space → pixel space)
# ------------------------------------------------------------------

def coord_to_pixel(x: float, y: float) -> tuple[int, int]:
    """
    Pyglet Y-up → PIL Y-down, and divide by SCALE to go from coordinate
    space to image pixels.
    """
    px = int(round(x / SCALE))
    py = int(round((COORD_H - y) / SCALE))
    return px, py


def ring_to_pixels(pts: list) -> list[tuple[int, int]]:
    return [coord_to_pixel(x, y) for x, y in pts]


def build_images(outer_pts, inner_pts, grass_outer_pts, grass_inner_pts,
                 centre_pts, gates):
    """Return (background_img, tarmac_img) as PIL RGBA images.

    background_img:
        Green grass fills the *expanded* outer polygon and uses the *shrunk*
        inner polygon as the transparent hole — so green overflows the road
        edge on BOTH the outside and the inside of the track.

    tarmac_img:
        Dark asphalt fills exactly the road band (outer minus inner).
    """
    bg   = Image.new('RGBA', (IMG_W, IMG_H), TRANSPARENT)
    tarm = Image.new('RGBA', (IMG_W, IMG_H), TRANSPARENT)

    bg_draw   = ImageDraw.Draw(bg)
    tarm_draw = ImageDraw.Draw(tarm)

    grass_outer_px = ring_to_pixels(grass_outer_pts)
    grass_inner_px = ring_to_pixels(grass_inner_pts)
    outer_px       = ring_to_pixels(outer_pts)
    inner_px       = ring_to_pixels(inner_pts)

    # Background: expanded outer grass fully filled (no hole) — the infield is
    # green, then tarmac.png is composited on top in Pyglet.
    # The grass_inner polygon is only used so tarmac doesn't bleed into the infield.
    bg_draw.polygon(grass_outer_px, fill=GRASS_COLOR)

    # Tarmac: road band only (outer boundary → transparent centre)
    tarm_draw.polygon(outer_px, fill=ROAD_COLOR)
    tarm_draw.polygon(inner_px,  fill=TRANSPARENT)

    # Kerb outlines on the road edges
    tarm_draw.polygon(outer_px, outline=(150, 150, 150, 220))
    tarm_draw.polygon(inner_px,  outline=(150, 150, 150, 220))

    # Dashed centre line
    centre_px = ring_to_pixels(resample(centre_pts, 10.0))
    for i in range(0, len(centre_px) - 1, 2):
        tarm_draw.line([centre_px[i], centre_px[(i + 1) % len(centre_px)]],
                       fill=LINE_COLOR, width=1)

    return bg, tarm


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def _compute_start(gates):
    """Derive start position and heading from gate 0 midpoint and gate 0→1 direction."""
    g0_mid = ((gates[0][0][0] + gates[0][1][0]) / 2,
              (gates[0][0][1] + gates[0][1][1]) / 2)
    g1_mid = ((gates[1][0][0] + gates[1][1][0]) / 2,
              (gates[1][0][1] + gates[1][1][1]) / 2)
    dx = g1_mid[0] - g0_mid[0]
    dy = g1_mid[1] - g0_mid[1]
    hdg = math.degrees(math.atan2(dx, dy)) % 360
    # place car just before gate 0 along the approach direction
    length = math.hypot(dx, dy) or 1.0
    sx = g0_mid[0] - (dx / length) * 26
    sy = g0_mid[1] - (dy / length) * 26
    return round(sx, 1), round(sy, 1), round(hdg, 1)


def main():
    parser = argparse.ArgumentParser(description='Generate track2 assets')
    parser.add_argument('--name', type=str, default='track2',
                        help='output track name (folder under assets/tracks/)')
    args = parser.parse_args()

    print(f'Generating {args.name}...')

    # output folder
    from pathlib import Path as _Path
    tracks_base = _Path(__file__).resolve().parent / 'assets' / 'tracks'
    out_dir = tracks_base / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build centreline
    raw_centre = build_centreline()
    smooth_centre = smooth_polygon(raw_centre, iterations=4)
    centre = resample(smooth_centre, spacing=8.0)
    print(f'  Centreline: {len(centre)} points')

    # 2. Build outer / inner rings
    outer_pts, inner_pts, grass_outer_pts, grass_inner_pts = build_rings(
        centre, half_width=TRACK_WIDTH / 2
    )
    print(f'  Outer ring: {len(outer_pts)} pts  Inner ring: {len(inner_pts)} pts')

    # 3. track.npy
    outer_segs = polygon_to_segments(outer_pts, spacing=12.0)
    inner_segs = polygon_to_segments(inner_pts, spacing=12.0)
    all_segs   = np.vstack([outer_segs, inner_segs])
    np.save(str(out_dir / 'track.npy'), all_segs)
    print(f'  Saved track.npy  ({len(all_segs)} segments: '
          f'{len(outer_segs)} outer + {len(inner_segs)} inner)')

    # 4. gates.npy
    gates = build_gates(centre, N_GATES, TRACK_WIDTH)
    np.save(str(out_dir / 'gates.npy'), gates)
    print(f'  Saved gates.npy  ({len(gates)} gates)')

    # 5. Images
    bg_img, tarm_img = build_images(outer_pts, inner_pts,
                                    grass_outer_pts, grass_inner_pts, centre, gates)
    bg_img.save(str(out_dir / 'background.png'))
    tarm_img.save(str(out_dir / 'tarmac.png'))
    print(f'  Saved images to {out_dir}/')

    # 6. meta.json
    sx, sy, sh = _compute_start(gates)
    from system.track_registry import write_meta
    write_meta(out_dir, sx, sy, sh, image_scale=1.3)
    print(f'  Saved meta.json  start=({sx},{sy},{sh}°)')

    # 7. Sanity check
    from shapely.geometry import Polygon as SPoly, LinearRing as SRing
    road = SPoly(SRing(outer_pts)).difference(SPoly(SRing(inner_pts)))
    print(f'  Road area: {road.area:.0f} sq px  valid={road.is_valid}')

    print('Done.')


if __name__ == '__main__':
    main()
