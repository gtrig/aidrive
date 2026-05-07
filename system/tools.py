import math
import numpy as np
from shapely.geometry import LineString,Point


class MapTools():
    def __init__(self, *args, **kwargs):
        self.track = []
        self.usedPoints = []
        self.unusedPoints = []
        self.lines = []
        self.firstPoint = []
        self.lastPoint = []

    def loadTrackArray(self,track):
        self.track = track
        self.unusedPoints = self.track
        self.firstPoint = self.unusedPoints.pop()
        self.lastPoint = self.firstPoint
        #self.usedPoints.append(self.firstPoint)
    

    def findClosestPoint(self,point):
        #print("looking for ", point,len(self.unusedPoints))
        
        min_dist = 100000000000
        
        for i,p in enumerate(self.unusedPoints):
            #distance = math.sqrt((p[0]-point[0])**2)+((p[1]-point[1])**2)
            distance = self.dist(p,point)
            if(min_dist>distance):
                candidate = p
                min_dist = min(min_dist,distance)
        return candidate

    def dist(self,p1,p2):
        return math.hypot(p1[1]-p2[1],p1[0]-p2[0])

    def create_line(self, p1,p2):
        self.lines.append([p1,p2])
    
    def usePoint(self,point):
        self.unusedPoints.remove(point)
        self.usedPoints.append(point)
        self.lastPoint = point

    def outlineTrack(self):
        while len(self.unusedPoints)>0:
            clPoint = self.findClosestPoint(self.lastPoint)
            #print("Creating line with size:",self.dist(self.lastPoint, clPoint))
            if(self.dist(self.lastPoint, clPoint)<40):
                self.create_line(self.lastPoint, clPoint)
            self.usePoint(clPoint)
        
        #self.create_line(self.lastPoint, self.firstPoint)
    
    def thinLines(self,lines):
        newLines = []
        for i in range(0,len(lines)-1,2):
            a = lines[i]
            b = lines[i+1]
            c = a[0],b[1]
            newLines.append(c)
        return newLines


class LineTools():
    def __init__(self,*args,**kwargs):
        self.sensors = kwargs.get('sensors', None)
        self.lines = kwargs.get('lines', None)
        self.poi_x = 0
        self.poi_y = 0
        self.poi_r = kwargs.get('radius', 70)
        self.poi_x_max = 0
        self.poi_x_min = 0
        self.poi_y_max = 0
        self.poi_y_min = 0
        self.linesSample = []

    def updatePOI(self,x,y):
        self.poi_x = x
        self.poi_y = y
        self.calculatePOIBox()

    def calculatePOIBox(self):
        self.poi_x_max = self.poi_x+self.poi_r
        self.poi_x_min = self.poi_x-self.poi_r
        self.poi_y_max = self.poi_y+self.poi_r
        self.poi_y_min = self.poi_y-self.poi_r
        

    def getLinesInBox(self):
        linesInside = []
        for line in self.lines:
            if(line[0][0]>self.poi_x_max) or (line[1][0]>self.poi_x_max) or (line[0][0]<self.poi_x_min) or (line[1][0]<self.poi_x_min) or (line[0][1]>self.poi_y_max) or (line[1][1]>self.poi_y_max) or (line[0][1]<self.poi_y_min) or (line[1][1]<self.poi_y_min):
                pass
            else:
                linesInside.append(line)
        self.linesSample = linesInside
        return linesInside

    def line_intersection(self, line1, line2):
        l1 = LineString([(line1[0][1],line1[0][0]), (line1[1][1],line1[1][0])])
        l2 = LineString([(line2[0][1],line2[0][0]), (line2[1][1],line2[1][0])])
        point = l2.intersection(l1)
        if point.geom_type == "Point":
            return [round(point.x,2) , round(point.y,2)]
        #
        #print(point.coords[:])
        # LineString(coords).crosses(LineString([(0, 1), (1, 0)]))
        #

    def distance(self,point1,point2):
        p1 = Point(point1[1],point1[0])
        p2 = Point(point2[0],point2[1])
        distance = p1.distance(p2)
        
        return distance
#FIX THIS: coordintates of line and intercept is not in same order!

    def getIntesections(self):

        intersections = []
        
        for sensor in self.sensors:
            #sensor.toString()
            leastdistance=1001
            for line in self.linesSample:
                intersection_point = self.line_intersection([sensor.p1,sensor.p2],line)
                if(intersection_point):
                    distance=self.distance(sensor.p1,intersection_point)
#                    if sensor.offset==180:
#                       print ("p1:",sensor.p1," p2:",sensor.p2," int:",intersection_point," dist:",distance)
                    if distance<leastdistance:
                        nearestIntersection=intersection_point
                        leastdistance=distance
 
            if leastdistance<1000:
                sensor.hit(round(leastdistance,2))
                intersections.append(nearestIntersection)
            else:
                sensor.reset()

        return intersections
