import unittest
import math
import numpy as np
from system.tools import MapTools, LineTools
from entities.car import sensor


class TestTools(unittest.TestCase):
    def test_loadArray(self):
        trackArray = [
            [1,0],
            [2,2],
            [5,4],
            [3,4],
            [3,5],
            [10,10]]

        t = MapTools()
        t.loadTrackArray(trackArray)

        self.assertEqual(t.track[2][1],4)
        self.assertEqual(t.track[3][1],4)
        self.assertEqual(t.track[3][0],3)
        #self.assertEquals(t.track[5][1],10)
        self.assertEqual(t.firstPoint,[10,10])
    
    def test_findClosest(self):
        trackArray = [
            [1,0],
            [2,2],
            [5,4],
            [3,4],
            [3,5],
            [10,10]]

        t = MapTools()
        t.loadTrackArray(trackArray)

        clPoint = t.findClosestPoint(t.firstPoint)

        self.assertEqual(clPoint,[5,4])
        t.usePoint(clPoint)
        self.assertEqual(t.usedPoints[len(t.usedPoints)-1],clPoint)
        t.create_line(t.firstPoint,clPoint)
        self.assertEqual(t.lines[0],[[10,10],[5,4]])

    def test_getLines(self):
        trackArray = [
            [1,0],
            [2,2],
            [5,4],
            [3,4],
            [3,5],
            [1,7],
            [3,8],
            [6,9],
            [7,2],
            [10,10]]

        t = MapTools()
        t.loadTrackArray(trackArray)
        t.outlineTrack()

        #print(t.lines)

    def test_sensors(self):
        sensors = []

        s = sensor(size=10,offset=0)
        s.p1 = [0,0]
        s.recalculatePoints()
        sensors.append(s)
        
        s = sensor(size=10,offset=90)
        s.p1 = [0,0]
        s.recalculatePoints()
        sensors.append(s)
        

        #sensors = [[[0,0],[10,0]],[[0,0],[0,10]]]
        lines = [[[8,-5],[2,5]]]
        
        t = LineTools(sensors=sensors,lines=lines)
        t.calculatePOIBox()
        t.getLinesInBox()
        print(t.getIntesections())


    # ------------------------------------------------------------------
    # Phase-1 parity tests: NumPy vectorised sensor intersection
    # ------------------------------------------------------------------

    def _make_linetools(self, sensor_specs, lines, poi_x=0, poi_y=0, radius=200):
        """Helper: build a LineTools, place sensors, run box filter."""
        sensors = []
        for size, offset in sensor_specs:
            s = sensor(size=size, offset=offset)
            s.p1 = [poi_x, poi_y]
            s.recalculatePoints(heading=0)
            sensors.append(s)
        lt = LineTools(sensors=sensors, lines=lines)
        lt.updatePOI(poi_x, poi_y)
        lt.getLinesInBox()
        return lt, sensors

    def test_intersection_straight_ahead(self):
        """Sensor pointing straight up (offset=0) hits a horizontal wall."""
        # Sensor: from (0,0) pointing up, size=100
        # Wall: horizontal line at y=50, x in [-20, 20]
        lt, sensors = self._make_linetools(
            [(100, 0)],
            [[[-20, 50], [20, 50]]],
        )
        lt.getIntesections()
        self.assertAlmostEqual(sensors[0].distance, 50.0, places=1)

    def test_intersection_right_sensor(self):
        """Sensor pointing right (offset=90) hits a vertical wall."""
        # Sensor: from (0,0) pointing right, size=100
        # Wall: vertical line at x=40, y in [-20, 20]
        lt, sensors = self._make_linetools(
            [(100, 90)],
            [[[40, -20], [40, 20]]],
        )
        lt.getIntesections()
        self.assertAlmostEqual(sensors[0].distance, 40.0, places=1)

    def test_no_intersection_resets_distance(self):
        """Sensor that misses all lines resets distance to 1000."""
        lt, sensors = self._make_linetools(
            [(100, 0)],
            [[[200, 200], [210, 200]]],  # wall far away, outside radius
        )
        lt.getIntesections()
        self.assertEqual(sensors[0].distance, 1000)

    def test_multiple_sensors_independent(self):
        """Two sensors each hit different walls; distances are independent."""
        # Sensor 0: offset=0  (up)   hits y=30
        # Sensor 1: offset=90 (right) hits x=60
        lt, sensors = self._make_linetools(
            [(200, 0), (200, 90)],
            [[[-50, 30], [50, 30]],   # horizontal at y=30
             [[60, -50], [60, 50]]],  # vertical at x=60
        )
        lt.getIntesections()
        self.assertAlmostEqual(sensors[0].distance, 30.0, places=1)
        self.assertAlmostEqual(sensors[1].distance, 60.0, places=1)

    def test_nearest_wall_wins(self):
        """When two walls cross the same ray, the closer one is used."""
        lt, sensors = self._make_linetools(
            [(200, 0)],
            [[[-10, 40], [10, 40]],   # closer at y=40
             [[-10, 80], [10, 80]]],  # further at y=80
        )
        lt.getIntesections()
        self.assertAlmostEqual(sensors[0].distance, 40.0, places=1)

    def test_getLinesInBox_filters_correctly(self):
        """Lines outside the POI radius are excluded from the sample."""
        lt, sensors = self._make_linetools(
            [(50, 0)],
            [[[-5, 10], [5, 10]],      # inside radius=200
             [[500, 500], [510, 500]]], # way outside
            radius=100,
        )
        # Only one line should survive the box filter
        self.assertEqual(len(lt.linesSample), 1)

    def test_seven_sensors_full_layout(self):
        """Training layout (7 sensors, mixed angles) should not raise or hang."""
        layout = [(-90, 20), (-45, 50), (-20, 80), (0, 100), (20, 80), (45, 50), (90, 20)]
        lines = [
            [[-30, 15], [30, 15]],
            [[15, -30], [15, 30]],
            [[-15, -30], [-15, 30]],
        ]
        lt, sensors = self._make_linetools(layout, lines)
        # Should not raise
        result = lt.getIntesections()
        # Every sensor should have a finite distance or be reset to 1000
        for s in sensors:
            self.assertGreater(s.distance, 0)
            self.assertLessEqual(s.distance, 1000)


if __name__ == '__main__':
    unittest.main()