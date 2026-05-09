"""
Track 3 generator.  Much more complex than track2.

Builds a self-contained track folder under assets/tracks/<name>/:
  background.png    (RGBA 1000x578)
  tarmac.png        (RGBA 1000x578)
  track.npy         (collision line segments)
  gates.npy         (54 gate checkpoints)
  meta.json         (start position + image scale)

Run:
    .venv/bin/python gen_track3.py
    .venv/bin/python gen_track3.py --name circuit_sigma
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import make_valid

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

IMG_W, IMG_H = 1000, 578
SCALE        = 1.3
COORD_W      = IMG_W * SCALE   # ~1300
COORD_H      = IMG_H * SCALE   # ~751

TRACK_WIDTH  = 68              # slightly narrower than track2 for tighter corners
GRASS_MARGIN = 20
N_GATES      = 54

GRASS_COLOR  = (72, 130, 40, 255)
ROAD_COLOR   = (55, 55, 55, 255)
TRANSPARENT  = (0, 0, 0, 0)
LINE_COLOR   = (255, 255, 255, 200)

# ------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------

def _arc(cx, cy, r, a0_deg, a1_deg, n=30):
    """CCW arc from a0_deg to a1_deg.  Returns list of (x,y) tuples."""
    a0 = math.radians(a0_deg)
    a1 = math.radians(a1_deg)
    # always go in increasing angle direction; wrap if needed
    if a1 <= a0:
        a1 += 2 * math.pi
    angles = np.linspace(a0, a1, max(n, 2), endpoint=False)
    return [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in angles]


def _line(x0, y0, x1, y1, n=8):
    xs = np.linspace(x0, x1, n, endpoint=False)
    ys = np.linspace(y0, y1, n, endpoint=False)
    return list(zip(xs, ys))


def smooth_polygon(pts, iterations=3):
    """Chaikin corner-cutting smoothing."""
    arr = np.array(pts, dtype=float)
    for _ in range(iterations):
        q = 0.75 * arr + 0.25 * np.roll(arr, -1, axis=0)
        r = 0.25 * arr + 0.75 * np.roll(arr, -1, axis=0)
        arr = np.empty((len(arr) * 2, 2))
        arr[0::2] = q
        arr[1::2] = r
    return list(map(tuple, arr))


def resample(pts, spacing=15.0):
    arr  = np.array(pts, dtype=float)
    arr  = np.vstack([arr, arr[0]])
    d    = np.diff(arr, axis=0)
    dist = np.hypot(d[:, 0], d[:, 1])
    cum  = np.concatenate([[0], np.cumsum(dist)])
    tgt  = np.arange(0, cum[-1], spacing)
    xs   = np.interp(tgt, cum, arr[:, 0])
    ys   = np.interp(tgt, cum, arr[:, 1])
    return list(zip(xs, ys))

# ------------------------------------------------------------------
# Track 3 centreline
# ------------------------------------------------------------------

def build_centreline():
    """
    Polar-Fourier centreline — guaranteed non-self-intersecting (star-convex).

    The circuit is described in polar coordinates as r(θ) around a centre point.
    Because r(θ) > 0 for every θ and is single-valued, the curve can never cross
    itself regardless of how many bumps and dips are added.

    Feature map (θ measured CCW from east):
      ~270° ( bottom  )  long flat section → main straight
      ~ 15°  (right)     Gaussian dip → tight hairpin 1
      ~ 60°  (upper-R)   Gaussian dip → tight hairpin 2
      ~120°  (upper-L)   chicane ripple
      ~195°  (left)      Gaussian dip → tight left hairpin
      ~240°  (lower-L)   chicane ripple
      ~310°  (lower-R)   shallow dip → technical corner
    """
    N = 800
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)

    # ── Polar radius ──────────────────────────────────────────────────
    r = np.full(N, 300.0)

    # Oval base: wider east–west than north–south
    r += 50 * np.cos(t + 0.4)
    r -= 22 * np.cos(2 * t + 0.2)

    # ── Tight hairpins (deep Gaussian dips toward the centre) ────────
    def dip(t0_deg, depth, width_deg):
        t0 = math.radians(t0_deg)
        w  = math.radians(width_deg)
        dt = np.arctan2(np.sin(t - t0), np.cos(t - t0))   # wrap to (−π, π)
        return -depth * np.exp(-0.5 * (dt / w) ** 2)

    r += dip(  5, depth=145, width_deg=14)   # hairpin 1  – far right
    r += dip( 58, depth=120, width_deg=12)   # hairpin 2  – upper-right
    r += dip(190, depth=130, width_deg=13)   # hairpin 3  – far left
    r += dip(310, depth= 80, width_deg=16)   # technical  – lower-right

    # ── Chicane complexes (sinusoidal ripple in a localised window) ──
    def chicane(t0_deg, amplitude, freq, width_deg):
        t0 = math.radians(t0_deg)
        w  = math.radians(width_deg)
        dt = np.arctan2(np.sin(t - t0), np.cos(t - t0))
        return amplitude * np.sin(freq * dt) * np.exp(-0.5 * (dt / w) ** 2)

    r += chicane(115, amplitude=35, freq=5, width_deg=30)   # triple chicane – upper-left
    r += chicane(250, amplitude=30, freq=5, width_deg=26)   # double chicane – lower-left

    # safety check
    assert r.min() > 10, f"r goes non-positive: min={r.min():.1f}"

    # ── Convert to Cartesian and scale to fill canvas ─────────────────
    # Coordinate space: ~1300 wide × 751 tall.  Leave ~100 px margin each side.
    cx, cy = 645, 380
    sx = 530 / r.max()     # normalise to desired half-width
    sy = 310 / r.max()     # normalise to desired half-height  (flatter)
    xs = cx + r * np.cos(t) * sx * 1.55
    ys = cy + r * np.sin(t) * sy * 1.55

    xs = np.clip(xs, 100, 1200)
    ys = np.clip(ys, 85,  665)

    return list(zip(xs, ys))


# ------------------------------------------------------------------
# Ring building
# ------------------------------------------------------------------

GRASS_MARGIN_OUTER = GRASS_MARGIN
GRASS_MARGIN_INNER = GRASS_MARGIN

def build_rings(centre, half_width):
    poly        = Polygon(centre)
    if not poly.is_valid:
        poly = make_valid(poly)
        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda g: g.area)

    outer       = poly.buffer(half_width,                      join_style=1)
    inner       = poly.buffer(-half_width,                     join_style=1)
    grass_outer = poly.buffer(half_width + GRASS_MARGIN_OUTER, join_style=1)

    def _coords(geom):
        if isinstance(geom, MultiPolygon):
            geom = max(geom.geoms, key=lambda g: g.area)
        return list(geom.exterior.coords)

    return _coords(outer), _coords(inner), _coords(grass_outer)


# ------------------------------------------------------------------
# .npy generation
# ------------------------------------------------------------------

def polygon_to_segments(pts, spacing=12.0):
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

def build_gates(centre_pts, n_gates, track_width):
    dense = resample(centre_pts, spacing=1.0)
    n     = len(dense)
    arr   = np.array(dense, dtype=float)
    idxs  = np.linspace(0, n - 1, n_gates, endpoint=False, dtype=int)
    gates = np.zeros((n_gates, 2, 2), dtype=np.int32)
    half  = track_width * 0.55
    for g, idx in enumerate(idxs):
        cx, cy = arr[idx]
        tx = arr[(idx+1) % n][0] - arr[(idx-1) % n][0]
        ty = arr[(idx+1) % n][1] - arr[(idx-1) % n][1]
        ln = math.hypot(tx, ty) or 1.0
        nx, ny = -ty/ln, tx/ln
        gates[g] = [
            [round(cx + nx*half), round(cy + ny*half)],
            [round(cx - nx*half), round(cy - ny*half)],
        ]
    return gates


# ------------------------------------------------------------------
# Image generation
# ------------------------------------------------------------------

def coord_to_pixel(x, y):
    return int(round(x / SCALE)), int(round((COORD_H - y) / SCALE))


def ring_to_pixels(pts):
    return [coord_to_pixel(x, y) for x, y in pts]


def build_images(outer_pts, inner_pts, grass_outer_pts, centre_pts):
    bg   = Image.new('RGBA', (IMG_W, IMG_H), TRANSPARENT)
    tarm = Image.new('RGBA', (IMG_W, IMG_H), TRANSPARENT)
    bgd  = ImageDraw.Draw(bg)
    tmd  = ImageDraw.Draw(tarm)

    grass_px = ring_to_pixels(grass_outer_pts)
    outer_px = ring_to_pixels(outer_pts)
    inner_px = ring_to_pixels(inner_pts)

    # background: solid green blob (infield + grass border)
    bgd.polygon(grass_px, fill=GRASS_COLOR)

    # tarmac: road band only
    tmd.polygon(outer_px, fill=ROAD_COLOR)
    tmd.polygon(inner_px, fill=TRANSPARENT)
    tmd.polygon(outer_px, outline=(140, 140, 140, 200))
    tmd.polygon(inner_px, outline=(140, 140, 140, 200))

    # dashed centre line
    cpx = ring_to_pixels(resample(centre_pts, 10.0))
    for i in range(0, len(cpx) - 1, 2):
        tmd.line([cpx[i], cpx[(i+1) % len(cpx)]], fill=LINE_COLOR, width=1)

    return bg, tarm


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def _compute_start(gates):
    """Derive start position and heading from gate 0→1 direction."""
    g0_mid = ((gates[0][0][0] + gates[0][1][0]) / 2,
              (gates[0][0][1] + gates[0][1][1]) / 2)
    g1_mid = ((gates[1][0][0] + gates[1][1][0]) / 2,
              (gates[1][0][1] + gates[1][1][1]) / 2)
    dx = g1_mid[0] - g0_mid[0]
    dy = g1_mid[1] - g0_mid[1]
    hdg = math.degrees(math.atan2(dx, dy)) % 360
    length = math.hypot(dx, dy) or 1.0
    sx = g0_mid[0] - (dx / length) * 26
    sy = g0_mid[1] - (dy / length) * 26
    return round(sx, 1), round(sy, 1), round(hdg, 1)


def main():
    parser = argparse.ArgumentParser(description='Generate track3 assets')
    parser.add_argument('--name', type=str, default='track3',
                        help='output track name (folder under assets/tracks/)')
    args = parser.parse_args()

    print(f'Generating {args.name}...')

    tracks_base = Path(__file__).resolve().parent / 'assets' / 'tracks'
    out_dir = tracks_base / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Centreline
    raw    = build_centreline()
    smo    = smooth_polygon(raw, iterations=4)
    centre = resample(smo, spacing=8.0)
    print(f'  Centreline: {len(centre)} points')

    # 2. Validity check
    raw_poly = Polygon(centre)
    print(f'  Raw polygon valid: {raw_poly.is_valid}')
    if not raw_poly.is_valid:
        from shapely.validation import explain_validity
        print(f'  Reason: {explain_validity(raw_poly)}')
        raw_poly = make_valid(raw_poly)
        if isinstance(raw_poly, MultiPolygon):
            raw_poly = max(raw_poly.geoms, key=lambda g: g.area)
        centre = list(raw_poly.exterior.coords)
        print(f'  Fixed centreline: {len(centre)} points')

    # 3. Rings
    outer_pts, inner_pts, grass_pts = build_rings(centre, TRACK_WIDTH / 2)
    road_poly = Polygon(outer_pts).difference(Polygon(inner_pts))
    print(f'  Outer: {len(outer_pts)} pts  Inner: {len(inner_pts)} pts')
    print(f'  Road area: {road_poly.area:.0f}  valid: {road_poly.is_valid}')

    # 4. track.npy
    outer_segs = polygon_to_segments(outer_pts, 12.0)
    inner_segs = polygon_to_segments(inner_pts, 12.0)
    all_segs   = np.vstack([outer_segs, inner_segs])
    np.save(str(out_dir / 'track.npy'), all_segs)
    print(f'  Saved track.npy  ({len(all_segs)} segs: {len(outer_segs)} outer + {len(inner_segs)} inner)')

    # 5. gates.npy
    gates = build_gates(centre, N_GATES, TRACK_WIDTH)
    np.save(str(out_dir / 'gates.npy'), gates)
    sx, sy, sh = _compute_start(gates)
    print(f'  Saved gates.npy  start=({sx},{sy},{sh}°)')

    # 6. Images
    bg_img, tarm_img = build_images(outer_pts, inner_pts, grass_pts, centre)
    bg_img.save(str(out_dir / 'background.png'))
    tarm_img.save(str(out_dir / 'tarmac.png'))
    print(f'  Saved images to {out_dir}/')

    # 7. meta.json
    from system.track_registry import write_meta
    write_meta(out_dir, sx, sy, sh, image_scale=1.3)
    print(f'  Saved meta.json')

    print('Done.')


if __name__ == '__main__':
    main()
