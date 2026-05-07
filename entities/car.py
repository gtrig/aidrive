import pyglet
from system.component import Component
import config
import math
from pathlib import Path


DEFAULT_SENSOR_LAYOUT = [(-90, 45), (90, 45)]
HEADLESS_CAR_WIDTH = 10
HEADLESS_CAR_HEIGHT = 18


class Car(Component):

    def __init__(self, *args, **kwargs):
        """
        Creates a sprite using a car image.

        Extra kwargs:
          headless (bool): skip pyglet image/sprite creation (for training).
          sensor_layout (list of (offset, size) tuples): sensor configuration.
        """
        super(Car, self).__init__(*args, **kwargs)
        self.speed = kwargs.get('speed', 0)
        self.maxspeed = kwargs.get('maxspeed', 5)
        self.minspeed = kwargs.get('minspeed', -1)
        self.orientation = kwargs.get('heading', 0)
        self.draw_sensors = kwargs.get('sensors', True)
        self.headless = kwargs.get('headless', False)
        self.steering = 0
        self.throttle = 0
        self.acceleration = 0

        if not self.headless:
            project_root = Path(__file__).resolve().parent.parent
            car_image_path = project_root / 'assets' / 'images' / 'Audi.png'
            self.car_image = pyglet.image.load(str(car_image_path))
            self.car_image.anchor_x = self.car_image.width // 2
            self.car_image.anchor_y = self.car_image.height // 2
            self.car_sprite = pyglet.sprite.Sprite(self.car_image, self.x, self.y)
            self.car_sprite.update(scale=0.15)
            self.width = self.car_sprite.width
            self.height = self.car_sprite.height
        else:
            self.width = HEADLESS_CAR_WIDTH
            self.height = HEADLESS_CAR_HEIGHT

        self.x_direction = 0
        self.y_direction = 1

        sensor_layout = kwargs.get('sensor_layout', DEFAULT_SENSOR_LAYOUT)
        self.sensors = [sensor(offset=off, size=sz) for off, sz in sensor_layout]

        self.updateSensors()

    def update_self(self):
        """
        Increments x and y value and updates position.
        Also ensures that the car does not leave the screen area by changing its axis direction
        :return:
        """

        if (self.acceleration==0) and (self.speed!=0):
            if self.speed > 0 :
                self.speed-=0.02
            elif self.speed < 0 :
                self.speed+=0.02

        self.speed+=self.acceleration

        self.speed = round(self.speed,2)
        if(self.speed > self.maxspeed):
            self.speed = self.maxspeed

        if(self.speed < self.minspeed):
            self.speed = self.minspeed

        self.orientation+=self.steering
        angle = math.radians(self.orientation)

        [self.y_direction,self.x_direction] = [self.speed * math.cos(angle), self.speed * math.sin(angle)]

        if not self.headless:
            self.car_sprite.update(rotation=self.orientation)

        if self.y < 0:
            self.y = 0
            self.speed = 0

        if self.y > config.window_height:
            self.y = config.window_height
            self.speed = 0

        if self.x < 0:
            self.x = 0 
            self.speed = 0

        if self.x > config.window_width:
            self.x = config.window_width
            self.speed=0

        if(self.speed > 0):
            self.x += (self.speed * self.x_direction)
            self.y += (self.speed * self.y_direction)
        else:
            self.x -= (self.speed * self.x_direction)
            self.y -= (self.speed * self.y_direction)
        if not self.headless:
            self.car_sprite.position = (self.x, self.y)
        self.updateSensors()

    def draw_self(self):
        """
        Draws our car sprite to screen
        :return:
        """
        if self.headless:
            return

        if self.draw_sensors:
            for sensor in self.sensors:
                self.draw_line(self.x, self.y, self.orientation + sensor.offset, sensor.size)

        self.car_sprite.draw()
        
        

    def accelerate(self,value):
        self.acceleration=value

    def turn(self,value):
        self.steering=value

    def draw_line(self,x,y,orientation,size):
        angle = math.radians(orientation)
        pyglet.graphics.draw(2, pyglet.gl.GL_LINES,
            ('v2f', (x, y, x+size*math.sin(angle), y+size*math.cos(angle)))
        )

    def updateSensors(self):
        #print('Sensor count:',len(self.sensors))
        for sensor in self.sensors:
            sensor.p1 = [self.x , self.y]
            sensor.recalculatePoints(self.orientation)

class sensor():
    def __init__(self, *args, **kwargs):
        self.offset = kwargs.get('offset', 0)
        self.size = kwargs.get('size',100)
        self.distance = 1000
        self.p1 = []
        self.p2 = []
    
    def recalculatePoints(self,heading=0):
        angle = math.radians(heading+self.offset)
        x = self.p1[0]
        y = self.p1[1]
        self.p2 = [round(x+self.size*math.sin(angle)), round(y+self.size*math.cos(angle))]

    def toString(self):
        print(self.p1,self.p2,self.size,self.offset)
    
    def reset(self):
        self.distance=1000

    def hit(self, distance):
        self.distance = distance