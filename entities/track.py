import pyglet
import config
import math
import numpy as np
from PIL import Image, ImageFilter
from system.tools import MapTools
from system.track_registry import load as load_track_meta
from shapely.geometry import LinearRing,Point,Polygon,MultiPolygon
from matplotlib import pyplot as plt,patches
from pathlib import Path

class Track():
    def __init__(self, *args, **kwargs):
        self.headless = kwargs.get('headless', False)
        track_id      = kwargs.get('track_id', 'track1')

        meta = load_track_meta(track_id)

        if not self.headless:
            self.border_image = pyglet.image.load(str(meta.background_png))
            self.tarmac_image = pyglet.image.load(str(meta.tarmac_png))
            self.border_sprite = pyglet.sprite.Sprite(self.border_image, 0, 0)
            self.tarmac_sprite = pyglet.sprite.Sprite(self.tarmac_image, 0, 0)
            self.border_sprite.update(scale=meta.image_scale)
            self.tarmac_sprite.update(scale=meta.image_scale)

        self.coords_map = []
        self.lines = np.load(str(meta.track_npy))
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
        if self.headless:
            return
        self.border_sprite.draw()
        self.tarmac_sprite.draw()
        # pyglet.graphics.draw(2, pyglet.gl.GL_LINES,
        # ('v2i', ([100, 150, 100,300 ]))
        # )
    
    def find_linerings(self):
        if self.linerings_found:
            return
        firstpoint = self.lines[0][0]
        points = []
        previousEnd = firstpoint[1]
        self.linerings_found = True
        for line in self.lines:
            if points == [] or (line[0] == previousEnd).all():
                points.append(line[1])
            else:
                if len(points) >= 3:
                    self.linerings.append(Polygon(LinearRing(points)))
                points = []
            previousEnd = line[1]

        # Flush the last ring before computing road so that tracks with only
        # one ring-break in the .npy file (e.g. track2) still work.
        if len(points) >= 3:
            self.linerings.append(Polygon(LinearRing(points)))

        road = self.linerings[0].difference(self.linerings[1])
        self.road = road

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
        if self.road.contains(Point(px, py)):
            return True
        else:
            return False
            
        for ring in self.linerings:
            self.n+=1
            if ring.contains(Point(px,py)):
                #print("Inside ring ",self.n)
                self.hits+=1
        if self.hits==0:
            pass
            #print("Outside")
        