import pyglet
import config
import math
import numpy as np
from PIL import Image, ImageFilter
from system.tools import MapTools
from shapely.geometry import LinearRing,Point,Polygon,MultiPolygon
from matplotlib import pyplot as plt,patches
from pathlib import Path

class Track():
    def __init__(self, *args, **kwargs):
        project_root = Path(__file__).resolve().parent.parent
        track_images_dir = project_root / 'assets' / 'images' / 'track1'
        self.border_image = pyglet.image.load(str(track_images_dir / 'background.png'))
        self.tarmac_image = pyglet.image.load(str(track_images_dir / 'tarmac.png'))
        self.border_sprite = pyglet.sprite.Sprite(self.border_image, 0, 0)
        self.tarmac_sprite = pyglet.sprite.Sprite(self.tarmac_image, 0, 0)
        self.border_sprite.update(scale=1.3)
        self.tarmac_sprite.update(scale=1.3)
        self.coords_map = []
        self.lines = np.load('track1.npy')
        self.linerings_found = False
        self.linerings = []
        self.find_linerings()


    
    def draw_outline(self):
        #draw coordsmap\
        #print(len(self.coords_map))
        
        for line in self.lines:
            pyglet.graphics.draw(2,pyglet.gl.GL_LINES,
                ('v2i', (line[0][0],line[0][1],line[1][0],line[1][1])),
                ('c3B', (255, 0, 0,255, 0, 0))
            )

        #     pyglet.graphics.draw(1,pyglet.gl.GL_POINTS,
        #         ('v2i', (line[0][0],line[0][1])),
        #         ('c3B', (255, 0, 0))
        #     )

        #     pyglet.graphics.draw(1,pyglet.gl.GL_POINTS,
        #         ('v2i', (line[1][0],line[1][1])),
        #         ('c3B', (255, 0, 0))
        #     )
        
        #print(*arr[250], sep = "\n") 




        # for point in self.coords_map:
        #     pyglet.graphics.draw(1,pyglet.gl.GL_POINTS,
        #         ('v2i', (point[0],point[1])),
        #         ('c3B', (255, 0, 0))
        #     )

        #Image.fromarray(arr).show()

        # for i in arr:
        #     for j in i:
        #         for k in j:
        #             if (arr[i][j][k]<127):
        #                 arr[i][j][k] = 0
        #             else:
        #                 arr[i][j][k] = 255

        # 
        

    def draw_self(self):
        self.border_sprite.draw()
        self.tarmac_sprite.draw()
        # pyglet.graphics.draw(2, pyglet.gl.GL_LINES,
        # ('v2i', ([100, 150, 100,300 ]))
        # )
    
    def find_linerings(self):
        if self.linerings_found:
            return
        # t = MapTools()
        firstpoint=self.lines[0][0]
        points=[]
        previousEnd =firstpoint[1]
        #print("First point:",firstpoint)
        self.linerings_found = True
        for line in self.lines:
            # print (line[0],line[1],line[0]==previousEnd,end='')
            if points==[] or (line[0]==previousEnd).all():
                points.append(line[1])
            else: 
                ring=Polygon(LinearRing(points))
                #print("Ring length:",len(ring.coords))
                #pyglet.graphics.draw(len(ring.coords), pyglet.gl.GL_LINE_LOOP,ring)
                self.linerings.append(ring)
                #self.plot(ring)
                #print(ring)
                points=[]

            previousEnd=line[1]

        road = self.linerings[0].difference(self.linerings[1])
        self.road = road
        
        
        #object.intersects(other)
#Returns True if the boundary or interior of the object intersect in any way with those of the other.

        if points!=[]:
            ring=Polygon(LinearRing(points))
            self.linerings.append(ring)

    def plot(self,polygon):
        x, y = polygon.exterior.coords.xy
        points = np.array([x, y], np.int32).T
        fig, ax = plt.subplots(1)
        polygon_shape = patches.Polygon(points, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(polygon_shape)
        plt.axis("auto")
        plt.show()



    def isInside(self, px, py):
        
        self.hits=0
        self.n=0
        if (self.road.contains(Point(px,py))):
            print ('in the road')
            return True
        else:
            print ('out of the road')
            return False
            
        for ring in self.linerings:
            self.n+=1
            if ring.contains(Point(px,py)):
                #print("Inside ring ",self.n)
                self.hits+=1
        if self.hits==0:
            pass
            #print("Outside")
        