from vpython import *
import numpy as np

import random


class world(box):
    def __init__(self):
        side = 4.0
        thk = 0.3
        s2 = 2 * side - thk
        s3 = 2 * side + thk
        self.wallR = box(
            pos=vector(side, 0, 0), size=vector(thk, s2, s3), color=color.red, opacity=0
        )
        self.wallL = box(
            pos=vector(-side, 0, 0),
            size=vector(thk, s2, s3),
            color=color.red,
            opacity=0,
        )
        self.wallB = box(
            pos=vector(0, -side, 0),
            size=vector(s3, thk, s3),
            color=color.blue,
            opacity=0,
        )
        self.wallT = box(
            pos=vector(0, side, 0),
            size=vector(s3, thk, s3),
            color=color.blue,
            opacity=0,
        )
        self.wallBK = box(
            pos=vector(0, 0, -side),
            size=vector(s2, s2, thk),
            color=color.gray(0.7),
            opacity=0,
        )


class fly(sphere):
    def __init__(self, side, thk, x, y, z, nudge_amt, blink_time):
        self.clock = random.uniform(0, 10)
        self.thk = thk
        self.blink_time = blink_time
        self.nudge_amt = nudge_amt
        self.ball = sphere(
            pos=vector(x, y, z), color=color.green, radius=0.1, retain=200
        )
        self.ball.mass = 1.0
        self.ball.p = vector(np.random.randn(), np.random.randn(), np.random.randn())

        self.side = side - self.thk * 0.5 - self.ball.radius

    def move(self, flys, radius, angle):
        dt = 0.3
        rate(200)
        self.ball.pos = self.ball.pos + (self.ball.p / self.ball.mass) * dt
        if not (self.side > self.ball.pos.x > -self.side):
            self.ball.p.x = -self.ball.p.x
        if not (self.side > self.ball.pos.y > -self.side):
            self.ball.p.y = -self.ball.p.y
        if not (self.side > self.ball.pos.z > -self.side):
            self.ball.p.z = -self.ball.p.z
        self.blink(flys, radius, angle)

    def blink(self, flys, radius, angle):
        num_neighbors = len(self.get_neighbors(flys, radius, angle))
        self.clock += 1
        if num_neighbors > 0:
            self.clock += self.clock * self.nudge_amt
        if self.clock > self.blink_time:
            self.ball.color = color.white
            self.clock = 0
        else:
            self.ball.color = color.green

    def get_neighbors(self, flys, radius, angle):
        """Return a list of neighbors within a field of view.

        boids: list of boids
        radius: field of view radius
        angle: field of view angle in radians

        returns: list of Boid
        """
        neighbors = []
        for fly in flys:
            if fly is self:
                continue
            offset = fly.ball.pos - self.ball.pos

            # if not in range, skip it
            if offset.mag > radius:
                continue

            # if not within viewing angle, skip it
            # diff = self.vel.diff_angle(offset)
            # if abs(diff) > angle:
            # continue

            # otherwise add it to the list
            neighbors.append(fly)

        return neighbors


fly_num = 500  # how many flys
nudge_amt = 0.5  # how much seeing a neighbor nudges time
blink_time = 15  # how many ticks per light up
sight_radius = 3  # sight radius of each fly
walls_thing = world()
flys = [
    fly(
        4.0,
        0.3,
        random.uniform(-3.5, 3.5),
        random.uniform(-3.5, 3.5),
        random.uniform(-3.5, 3.5),
        nudge_amt,
        blink_time,
    )
    for i in range(fly_num)
]

for i in range(1000):
    for fly in flys:
        fly.move(flys, sight_radius, 0)
