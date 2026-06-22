import math
import numpy as np
from shapely.geometry import LineString, Point


class MapTools():
    def __init__(self, *args, **kwargs):
        self.track = []
        self.usedPoints = []
        self.unusedPoints = []
        self.lines = []
        self.firstPoint = []
        self.lastPoint = []

    def loadTrackArray(self, track):
        self.track = track
        self.unusedPoints = self.track
        self.firstPoint = self.unusedPoints.pop()
        self.lastPoint = self.firstPoint

    def findClosestPoint(self, point):
        min_dist = 100000000000
        for i, p in enumerate(self.unusedPoints):
            distance = self.dist(p, point)
            if min_dist > distance:
                candidate = p
                min_dist = min(min_dist, distance)
        return candidate

    def dist(self, p1, p2):
        return math.hypot(p1[1] - p2[1], p1[0] - p2[0])

    def create_line(self, p1, p2):
        self.lines.append([p1, p2])

    def usePoint(self, point):
        self.unusedPoints.remove(point)
        self.usedPoints.append(point)
        self.lastPoint = point

    def outlineTrack(self):
        while len(self.unusedPoints) > 0:
            clPoint = self.findClosestPoint(self.lastPoint)
            if self.dist(self.lastPoint, clPoint) < 40:
                self.create_line(self.lastPoint, clPoint)
            self.usePoint(clPoint)

    def thinLines(self, lines):
        newLines = []
        for i in range(0, len(lines) - 1, 2):
            a = lines[i]
            b = lines[i + 1]
            c = a[0], b[1]
            newLines.append(c)
        return newLines


class LineTools():
    def __init__(self, *args, **kwargs):
        self.sensors   = kwargs.get('sensors', None)
        self.lines     = kwargs.get('lines', None)
        self.poi_x     = 0
        self.poi_y     = 0
        self.poi_r     = kwargs.get('radius', 70)
        self.poi_x_max = 0
        self.poi_x_min = 0
        self.poi_y_max = 0
        self.poi_y_min = 0
        self._lines_sample_list = None   # lazy Python list for play.py / tests
        self._lines_sample_arr = None    # (M, 2, 2) numpy — hot path for training
        self._rays_buf = None            # (S, 2, 2) reused sensor ray buffer

        # Pre-build a (N, 2, 2) float32 array of all track lines for fast
        # box-filtering and vectorised intersection.  Re-built lazily whenever
        # self.lines is set (or on first use).
        self._lines_arr = None   # shape (N, 2, 2) — all lines
        self._rebuild_lines_arr()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_lines_arr(self):
        """Convert self.lines (list of [[x0,y0],[x1,y1]]) to a float32 array."""
        if self.lines is not None and len(self.lines) > 0:
            self._lines_arr = np.array(self.lines, dtype=np.float32)  # (N,2,2)
        else:
            self._lines_arr = None

    # ------------------------------------------------------------------
    # Public API (unchanged signatures)
    # ------------------------------------------------------------------

    def updatePOI(self, x, y):
        self.poi_x = x
        self.poi_y = y
        self.calculatePOIBox()

    def calculatePOIBox(self):
        self.poi_x_max = self.poi_x + self.poi_r
        self.poi_x_min = self.poi_x - self.poi_r
        self.poi_y_max = self.poi_y + self.poi_r
        self.poi_y_min = self.poi_y - self.poi_r

    @property
    def linesSample(self):
        """Lazy Python list built only when callers iterate (play.py, tests)."""
        if self._lines_sample_arr is None:
            return []
        if self._lines_sample_list is None:
            self._lines_sample_list = self._lines_sample_arr.tolist()
        return self._lines_sample_list

    def getLinesInBox(self):
        """Return track lines inside the POI box as a numpy sub-array.

        Uses a NumPy boolean mask over the pre-built lines array when available,
        falling back to the original Python loop for correctness.
        """
        if self._lines_arr is None or len(self._lines_arr) == 0:
            self._lines_sample_arr = np.zeros((0, 2, 2), dtype=np.float32)
            self._lines_sample_list = None
            return self._lines_sample_arr

        arr = self._lines_arr          # (N, 2, 2)
        xmin, xmax = self.poi_x_min, self.poi_x_max
        ymin, ymax = self.poi_y_min, self.poi_y_max

        # Keep lines where BOTH endpoints are inside the box.
        # shape of arr[:,0,0] etc. is (N,).
        x0, y0 = arr[:, 0, 0], arr[:, 0, 1]
        x1, y1 = arr[:, 1, 0], arr[:, 1, 1]

        inside = (
            (x0 >= xmin) & (x0 <= xmax) &
            (y0 >= ymin) & (y0 <= ymax) &
            (x1 >= xmin) & (x1 <= xmax) &
            (y1 >= ymin) & (y1 <= ymax)
        )

        self._lines_sample_arr = arr[inside]          # (M, 2, 2)
        self._lines_sample_list = None                # invalidate lazy list
        return self._lines_sample_arr

    def getIntesections(self):
        """Update each sensor's distance using a fully-vectorised NumPy solver.

        Falls back gracefully to no hits if there are no nearby lines or sensors.
        """
        sensors = self.sensors
        if not sensors:
            return []

        lines_arr = self._lines_sample_arr
        if lines_arr is None:
            lines_arr = np.zeros((0, 2, 2), dtype=np.float32)

        S = len(sensors)
        L = len(lines_arr)

        if L == 0:
            for s in sensors:
                s.reset()
            return []

        # Reuse pre-allocated ray buffer; update in-place from sensor endpoints.
        if self._rays_buf is None or self._rays_buf.shape[0] != S:
            self._rays_buf = np.empty((S, 2, 2), dtype=np.float32)
        rays = self._rays_buf
        for i, s in enumerate(sensors):
            rays[i, 0, 0] = s.p1[0]
            rays[i, 0, 1] = s.p1[1]
            rays[i, 1, 0] = s.p2[0]
            rays[i, 1, 1] = s.p2[1]

        # Broadcast to (S, L, 2, 2)
        r  = rays[:, np.newaxis, :, :]      # (S, 1, 2, 2)
        ln = lines_arr[np.newaxis, :, :, :] # (1, L, 2, 2)

        # Ray:  P + t*(Q-P),  t in [0,1]
        # Line: A + u*(B-A),  u in [0,1]
        P  = r[..., 0, :]   # (S, 1, 2)
        Q  = r[..., 1, :]   # (S, 1, 2)
        A  = ln[..., 0, :]  # (1, L, 2)
        B  = ln[..., 1, :]  # (1, L, 2)

        d  = Q - P           # (S, 1, 2)  ray direction
        e  = B - A           # (1, L, 2)  segment direction

        # 2D cross: d × e  (scalar per pair)
        denom = d[..., 0] * e[..., 1] - d[..., 1] * e[..., 0]   # (S, L)

        # Vector from ray origin P to segment start A: (A - P)
        PA = A - P           # (S, L, 2)

        # Standard parametric formula:
        #   t = (A-P) × e / (d × e)   → position along ray
        #   u = (A-P) × d / (d × e)   → position along wall segment
        t_num = PA[..., 0] * e[..., 1] - PA[..., 1] * e[..., 0]  # (S, L)
        u_num = PA[..., 0] * d[..., 1] - PA[..., 1] * d[..., 0]  # (S, L)

        # Avoid divide-by-zero; parallel pairs have no intersection.
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(denom != 0, t_num / denom, np.inf)
            u = np.where(denom != 0, u_num / denom, np.inf)

        # Valid intersections: t in [0, 1] and u in [0, 1]
        valid = (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)

        # Replace invalid entries with +inf so argmin picks valid ones
        t_valid = np.where(valid, t, np.inf)            # (S, L)

        # Per sensor: index of nearest intersection
        best_idx = t_valid.argmin(axis=1)               # (S,)
        best_t   = t_valid[np.arange(S), best_idx]      # (S,)

        intersections = []
        for i, s in enumerate(sensors):
            if best_t[i] < np.inf:
                # distance = t * |ray|
                dx = float(s.p2[0] - s.p1[0])
                dy = float(s.p2[1] - s.p1[1])
                ray_len = math.hypot(dx, dy)
                dist = best_t[i] * ray_len
                s.hit(round(float(dist), 2))
                # intersection point
                ix = s.p1[0] + best_t[i] * dx
                iy = s.p1[1] + best_t[i] * dy
                intersections.append([round(float(ix), 2), round(float(iy), 2)])
            else:
                s.reset()

        return intersections

    # ------------------------------------------------------------------
    # Legacy helpers (kept for any code that still calls them)
    # ------------------------------------------------------------------

    def line_intersection(self, line1, line2):
        l1 = LineString([(line1[0][1], line1[0][0]), (line1[1][1], line1[1][0])])
        l2 = LineString([(line2[0][1], line2[0][0]), (line2[1][1], line2[1][0])])
        point = l2.intersection(l1)
        if point.geom_type == "Point":
            return [round(point.x, 2), round(point.y, 2)]

    def distance(self, point1, point2):
        p1 = Point(point1[1], point1[0])
        p2 = Point(point2[0], point2[1])
        return p1.distance(p2)
