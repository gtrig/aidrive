from system.tools import MapTools
import numpy as np
import pyglet
from PIL import Image, ImageFilter
from pathlib import Path

class map_gen():

    def load_track(self):
        self.scale = 1.3
        project_root = Path(__file__).resolve().parent
        track_images_dir = project_root / 'assets' / 'images' / 'track1'
        self.border_image = pyglet.image.load(str(track_images_dir / 'background.png'))
        self.tarmac_image = pyglet.image.load(str(track_images_dir / 'tarmac.png'))
        self.border_sprite = pyglet.sprite.Sprite(self.border_image, 0, 0)
        self.tarmac_sprite = pyglet.sprite.Sprite(self.tarmac_image, 0, 0)
        self.border_sprite.update(scale=self.scale)
        self.tarmac_sprite.update(scale=self.scale)
        self.coords_map = []
        self.lines = []

        img = Image.open(track_images_dir / 'trackmask.jpg')
        
        img = img.resize((self.tarmac_sprite.width, self.tarmac_sprite.height))
        edges = img.filter(ImageFilter.FIND_EDGES)
        arr = np.array(edges)
    
        point_map = np.empty([self.tarmac_sprite.height,self.tarmac_sprite.width])

        drawCount = 0

        for i,row in enumerate(arr):
            if (i == 0) or (i == len(arr)-1):
                for j,pixel in enumerate(row): #Reset the first and the last row of the image
                    arr[i][j] = [0,0,0]
                    point_map[i][j] = 0
            for j,pixel in enumerate(row):
                if (j == 0) or (j == len(row)-1): #reset the first and last pixels of 
                    arr[i][j] = [0,0,0]
                    point_map[i][j] = 0
                else:
                    if(np.sum(pixel)<250): # if there is something less than half in the pixel make it disappear
                        arr[i][j] = [0,0,0]
                        point_map[i][j] = 0
                    else:                   #else make it a point
                        arr[i][j] = [255,255,255]
                        point_map[i][j] = 1
                        
                        # self.coords_map.append([j,self.tarmac_sprite.height-i])    

                        if(drawCount>5):       #collect thined out points and append them to the coords_map
                            drawCount = 0
                            self.coords_map.append([j,self.tarmac_sprite.height-i])
                        else:
                            drawCount+=1
                    
        
        t = MapTools()
        t.loadTrackArray(self.coords_map)
        t.outlineTrack()
        self.lines = t.lines
        self.lines = t.thinLines(self.lines)
        self.lines = t.thinLines(self.lines)
        self.lines[205] = [[986,57],[1014,57]]
        self.lines.append([[998,124],[970,124]])
        #self.lines = t.thinLines(self.lines)


        #for i,line in enumerate(self.lines):
        #    print(i,line,t.dist(line[0],line[1]))
        
        np.save('track1.npy',self.lines)

        #print(self.lines)

if __name__ == '__main__' :
    m = map_gen()
    m.load_track()
    