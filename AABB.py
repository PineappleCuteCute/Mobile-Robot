import pygame
import numpy as np
from Colors import *

class AABB:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def return_coordinate(self):  # return (top_left.x, bottom_right.x, top_left.y, bottom_right.y)
        return self.x - self.width / 2, self.x + self.width / 2, self.y - self.height / 2, self.y + self.height / 2

    def get_corners(self):
        return [(self.x - self.width / 2, self.y - self.height / 2), (self.x + self.width / 2, self.y - self.height / 2),
                (self.x - self.width / 2, self.y + self.height / 2), (self.x + self.width / 2, self.y + self.height / 2)]

    def get_area(self):
        return self.width * self.height

    def get_intersect_area(self, otherAABB):
        x1, x2, y1, y2 = self.return_coordinate()
        x_1, x_2, y_1, y_2 = otherAABB.return_coordinate()
        top_left_x = max(x1, x_1)
        top_left_y = max(y1, y_1)
        bottom_right_x = min(x2, x_2)
        bottom_right_y = min(y2, y_2)
        return max(0, bottom_right_x - top_left_x) * max(0, bottom_right_y - top_left_y)

    def get_intersect_percentage(self, otherAABB):
        return self.get_intersect_area(otherAABB) / self.get_area()

    def get_quadrant(self, point):
        point_x, point_y = point
        if point_x <= self.x:
            if point_y <= self.y:
                return self.NW
            else:
                return self.SW
        else:
            if point_y <= self.y:
                return self.NE
            else:
                return self.SE

    def draw(self, window):
        pygame.draw.rect(window, BLACK, (self.x - self.width / 2, self.y - self.height / 2, self.width, self.height), 1)


class Obstacle(AABB):
    def __init__(self, x, y, width, height, static, v, x_bound=(20, 20), y_bound=(20, 20)):
        super().__init__(x, y, width, height)
        self.static = static
        self.v = np.array(v)
        if self.static:
            self.v = np.array([0, 0])
        self.x_bound = (x - x_bound[0], x + x_bound[1])
        self.y_bound = (y - y_bound[0], y + y_bound[1])
        self.counter = 0
        self.history = []

    def draw(self, window, with_past=True):
        if with_past:
            for pos_x, pos_y in self.history[-5:]:
                pygame.draw.rect(window, GREY, (pos_x - self.width / 2, pos_y - self.height / 2, self.width, self.height))
        color = BLACK if self.static else CYAN
        pygame.draw.rect(window, color, (self.x - self.width / 2, self.y - self.height / 2, self.width, self.height))

    def move(self):
        if not self.static:
            v_x, v_y = self.v
            self.x += v_x
            if self.x < self.x_bound[0]:
                self.x = self.x_bound[0]
                self.v = (-v_x, v_y)
            elif self.x > self.x_bound[1]:
                self.x = self.x_bound[1]
                self.v = (-v_x, v_y)
            self.y += v_y
            if self.y < self.y_bound[0]:
                self.y = self.y_bound[0]
                self.v = (v_x, -v_y)
            elif self.y > self.y_bound[1]:
                self.y = self.y_bound[1]
                self.v = (v_x, -v_y)
        # self.counter += 1
        # if self.counter % 2 == 0:
        self.history.append((self.x, self.y))

    def __str__(self):
        return f"Obstacle({self.x}, {self.y}, {self.width}, {self.height}, {self.static}, [{self.v[0]}, {self.v[1]}])"


class Node(AABB):
    def __init__(self, x, y, width, height, parent=None, region=None):
        super().__init__(x, y, width, height)
        # Neighbor list
        self.neighbors = []

        # D* properties
        self.in_queue = False
        self.h = 0
        self.rhs = np.inf
        self.g = np.inf
        self.key = [np.inf, np.inf]

        # Determining neighbors and values for QuadTree
        self.parent = parent
        self.percentage = 0
        self.value = -1
        self.NW = None
        self.NE = None
        self.SW = None
        self.SE = None
        self.region = region

        # Start / Goal indicators
        self.start = False
        self.goal = False

    def get_children(self):
        if self.NW is None:
            return None
        return [self.NW, self.NE, self.SW, self.SE]

    def split(self):
        self.NW = Node(self.x - self.width / 4, self.y - self.height / 4, self.width / 2, self.height / 2,
                       parent=self, region="NW")
        self.NE = Node(self.x + self.width / 4, self.y - self.height / 4, self.width / 2, self.height / 2,
                       parent=self, region="NE")
        self.SW = Node(self.x - self.width / 4, self.y + self.height / 4, self.width / 2, self.height / 2,
                       parent=self, region="SW")
        self.SE = Node(self.x + self.width / 4, self.y + self.height / 4, self.width / 2, self.height / 2,
                       parent=self, region="SE")

    def set_value(self, threshold_percentage, threshold_size):
        if self.width > threshold_size and self.height > threshold_size:
            if self.percentage <= threshold_percentage:
                self.value = 0
            elif self.percentage >= 1 - threshold_percentage:
                self.value = 1
                self.reset_children()
            else:
                self.value = -1
        else:
            if self.percentage <= threshold_percentage:
                self.value = 0
            else:
                self.value = 1

    def update_percentage_and_split(self, obstacles, threshold_percentage=0.005, threshold_size=16):
        for obstacle in obstacles:
            intersect_percentage = self.get_intersect_percentage(obstacle)
            self.percentage += intersect_percentage
        self.set_value(threshold_percentage, threshold_size)
        if self.value == -1:
            if self.get_children() is None:
                self.split()
            for child in self.get_children():
                child.update_percentage_and_split(obstacles, threshold_percentage, threshold_size)

    def get_north_neighbor(self):
        if self.parent is None:
            return None
        if self.region == "SW":
            return self.parent.NW
        if self.region == "SE":
            return self.parent.NE

        u = self.parent.get_north_neighbor()
        if u is None or u.get_children() is None:
            return u
        if self.region == "NW":
            return u.SW
        else:
            return u.SE

    def get_south_neighbor(self):
        if self.parent is None:
            return None
        if self.region == "NW":
            return self.parent.SW
        if self.region == "NE":
            return self.parent.SE

        u = self.parent.get_south_neighbor()
        if u is None or u.get_children() is None:
            return u
        if self.region == "SW":
            return u.NW
        else:
            return u.NE

    def get_left_neighbor(self):
        if self.parent is None:
            return None
        if self.region == "NE":
            return self.parent.NW
        if self.region == "SE":
            return self.parent.SW

        u = self.parent.get_left_neighbor()
        if u is None or u.get_children() is None:
            return u
        if self.region == "NW":
            return u.NE
        else:
            return u.SE

    def get_right_neighbor(self):
        if self.parent is None:
            return None
        if self.region == "NW":
            return self.parent.NE
        if self.region == "SW":
            return self.parent.SE

        u = self.parent.get_right_neighbor()
        if u is None or u.get_children() is None:
            return u
        if self.region == "NE":
            return u.NW
        else:
            return u.SW

    def get_north_west_neighbor(self):
        if self.parent is None:
            return None
        if self.region == "SE":
            return self.parent.NW
        if self.region == "NE":
            u = self.parent.get_north_neighbor()
            if u is None or u.get_children() is None:
                return u
            return u.SW
        if self.region == "SW":
            u = self.parent.get_left_neighbor()
            if u is None or u.get_children() is None:
                return u
            return u.NE
        u = self.parent.get_north_west_neighbor()
        if u is None or u.get_children() is None:
            return u
        return u.SE

    def get_north_east_neighbor(self):
        if self.parent is None:
            return None
        if self.region == "SW":
            return self.parent.NE
        if self.region == "NW":
            u = self.parent.get_north_neighbor()
            if u is None or u.get_children() is None:
                return u
            return u.SE
        if self.region == "SE":
            u = self.parent.get_right_neighbor()
            if u is None or u.get_children() is None:
                return u
            return u.NW
        u = self.parent.get_north_east_neighbor()
        if u is None or u.get_children() is None:
            return u
        return u.SW

    def get_south_west_neighbor(self):
        if self.parent is None:
            return None
        if self.region == "NE":
            return self.parent.SW
        if self.region == "SE":
            u = self.parent.get_south_neighbor()
            if u is None or u.get_children():
                return u
            return u.NW
        if self.region == "NW":
            u = self.parent.get_left_neighbor()
            if u is None or u.get_children() is None:
                return u
            return u.SE
        u = self.parent.get_south_west_neighbor()
        if u is None or u.get_children() is None:
            return u
        return u.NE

    def get_south_east_neighbor(self):
        if self.parent is None:
            return None
        if self.region == "NW":
            return self.parent.SE
        if self.region == "NE":
            u = self.parent.get_right_neighbor()
            if u is None or u.get_children() is None:
                return u
            return u.SW
        if self.region == "SW":
            u = self.parent.get_south_neighbor()
            if u is None or u.get_children() is None:
                return u
            return u.NE
        u = self.parent.get_south_east_neighbor()
        if u is None or u.get_children() is None:
            return u
        return u.NW

    def get_north_children(self):
        if self.get_children():
            return self.NW.get_north_children() + self.NE.get_north_children()
        return [self]

    def get_south_children(self):
        if self.get_children():
            return self.SW.get_south_children() + self.SE.get_south_children()
        return [self]

    def get_left_children(self):
        if self.get_children():
            return self.NW.get_left_children() + self.SW.get_left_children()
        return [self]

    def get_right_children(self):
        if self.get_children():
            return self.NE.get_right_children() + self.SE.get_right_children()
        return [self]

    def get_north_west_children(self):
        if self.get_children():
            return self.NW.get_north_west_children()
        return [self]

    def get_north_east_children(self):
        if self.get_children():
            return self.NE.get_north_east_children()
        return [self]

    def get_south_west_children(self):
        if self.get_children():
            return self.SW.get_south_west_children()
        return [self]

    def get_south_east_children(self):
        if self.get_children():
            return self.SE.get_south_east_children()
        return [self]

    def update_neighbors(self):
        north = self.get_north_neighbor()
        if north:
            self.neighbors = self.neighbors + north.get_south_children()
        south = self.get_south_neighbor()
        if south:
            self.neighbors = self.neighbors + south.get_north_children()
        left = self.get_left_neighbor()
        if left:
            self.neighbors = self.neighbors + left.get_right_children()
        right = self.get_right_neighbor()
        if right:
            self.neighbors = self.neighbors + right.get_left_children()
        north_west = self.get_north_west_neighbor()
        if north_west:
            self.neighbors = self.neighbors + north_west.get_south_east_children()
        north_east = self.get_north_east_neighbor()
        if north_east:
            self.neighbors = self.neighbors + north_east.get_south_west_children()
        south_west = self.get_south_west_neighbor()
        if south_west:
            self.neighbors = self.neighbors + south_west.get_north_east_children()
        south_east = self.get_south_east_neighbor()
        if south_east:
            self.neighbors = self.neighbors + south_east.get_north_west_children()

    def reset_children(self):
        self.NW = None
        self.NE = None
        self.SW = None
        self.SE = None

    def get_leaves(self):
        if self.get_children() is None:
            return [self]
        return self.NW.get_leaves() + self.NE.get_leaves() + self.SW.get_leaves() + self.SE.get_leaves()

    def calculate_key(self):
        self.key = [min(self.g, self.rhs) + self.h, min(self.g, self.rhs)]
        return self.key

    def calculate_rhs(self):
        ans = min([(cost(self, neighbor) + neighbor.g) for neighbor in self.neighbors])
        self.rhs = ans
        return ans

    def draw(self, window):
        pygame.draw.rect(window, BLACK, (self.x - self.width / 2, self.y - self.height / 2,
                                         self.width, self.height), 1 - self.value)


def distance(node1, node2):
    return np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)


def cost(node1, node2):
    return distance(node1, node2) / (1 + 1e-6 - min(1, node1.value * 100)) * (1 + 1e-6 - min(1, 100 * node2.value))
