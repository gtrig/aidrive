import pyglet
import config
from system.component import Component
from entities.car import Car
from entities.track import Track
from pyglet.window import key, FPSDisplay
from random import randint, choice
from system.tools import LineTools
import numpy as np
car = Car(x=500, y=90, speed=0, maxspeed=4, heading=270,sensors=True)
track = Track()
detectedlines = LineTools(sensors = car.sensors, lines = track.lines)
gates = np.load('gates.npy')
class Window(pyglet.window.Window):
    def __init_(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.set_minimum_size(300,200)


def draw():
    window.clear()
    track.draw_self()
    #track.draw_outline()
    
    
    dlines = detectedlines.getLinesInBox()
    for line in dlines:
        pyglet.graphics.draw(2,pyglet.gl.GL_LINES,
            ('v2i', (line[0][0],line[0][1],line[1][0],line[1][1])),
            ('c3B', (255, 0, 0,255, 0, 0))
        )

    for gate in gates:
        pyglet.graphics.draw(2,pyglet.gl.GL_LINES,
            ('v2f', (gate[0][0],gate[0][1],gate[1][0],gate[1][1])),
            ('c3B', (0, 255, 0, 0, 255, 0))
        )
    #print(len(dlines))

    intersections = detectedlines.getIntesections()
    count = 0
    for s in car.sensors:
        count+=1
        text = '{}:{}'.format(s.offset,s.distance)
        label = pyglet.text.Label(text,
                          font_name='Times New Roman',
                          font_size=10,
                          x=30, y=window.height-10-(15*count),
                          anchor_x='left', anchor_y='center')
    #print(len(intersections))
        label.draw()

    for ipoint in intersections:
        
         pyglet.graphics.draw(5,pyglet.gl.GL_POINTS,
            ('v2f', (
            ipoint[1],ipoint[0],
            ipoint[1]+1,ipoint[0],
            ipoint[1],ipoint[0]+1,
            ipoint[1]-1,ipoint[0],
            ipoint[1],ipoint[0]-1)),
            ('c3B', (0, 255, 0,0, 255, 0,0, 255, 0,0, 255, 0,0, 255, 0))
        )
    # print(car.x,car.y)

    car.draw_self()
    

def update(time):
    if isinstance(car, Component):
        car.update_self()
        track.isInside(car.x,car.y)
        detectedlines.updatePOI(car.x,car.y)

if __name__ == '__main__' :

    window = pyglet.window.Window(width=config.window_width,height=config.window_height,caption='pyCar',resizable=False)

    @window.event
    def on_draw():
        draw()

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == key.UP:
            car.accelerate(0.05)
        elif symbol == key.LEFT:
            car.turn(-5)
        elif symbol == key.RIGHT:
            car.turn(5)
        elif symbol == key.DOWN:
            car.accelerate(-0.1)

        # elif symbol == key.SPACE:
        #     s1 = car.sensors[0]
        #     s2 = car.sensors[1]
        #     s1.recalculatePoints(car.orientation)
        #     s2.recalculatePoints(car.orientation)
        #     gates.append([s1.p2,s2.p2])
        #     print(gates)
        # elif symbol == key.P:
        #     np.save('gates.npy',gates)
        # elif symbol == key.C:
        #     gates.pop()
        # elif symbol == key.V:
        #     print(gates)
        # # elif symbol == key.L:
        # #     gates = np.load('gates.npy')
        

    @window.event
    def on_key_release(symbol, modifiers):
        if symbol == key.UP:
            car.accelerate(0)
        elif symbol == key.LEFT:
            car.turn(0)
        elif symbol == key.RIGHT:
            car.turn(0)
        elif symbol == key.DOWN:
            car.accelerate(0)

    
    pyglet.clock.schedule_interval(update, 1/60.0)
    pyglet.app.run()