import numpy as np
import pygame
from DecisionMaking import DecisionMaking
from Solver import PriorityQueueSolver
from Colors import *

class Robot:
    def __init__(self, start: tuple, solver: PriorityQueueSolver, decisionMaker: DecisionMaking, r=40):
        self.pos = start
        self.r = r
        self.solver = solver
        self.decisionMaker = decisionMaker

    # def move(self, goal, obs):
    #     obstacles = self.detect(obs)
    #     self.pos = tuple(PSO(50, 50, self.pos, goal, obstacles=obstacles))

    def draw(self, window, draw_sr=True):
        if draw_sr:
            # draw the sensor range
            pygame.draw.circle(window,
                             RED,
                             self.pos,
                             self.r,
                             1)
        # draw the robot
        pygame.draw.circle(window, RED, self.pos, 8, 0)
        
    def reach(self, goal, epsilon=8):
        robotX, robotY = self.pos
        goalX, goalY = goal
        return ((robotX - goalX) ** 2 + (robotY - goalY) ** 2) <= epsilon ** 2

    def enter(self, node):
        robotX, robotY = self.pos
        nodeX, nodeY, node_size = node.x, node.y, node.width
        return np.abs(robotX - nodeX) <= node_size / 2 and np.abs(robotY - nodeY) <= node_size / 2

    def updatePath(self, obstacles_list):
        obstacles = self.detect(obstacles_list)
        # for n in env.current.neighbors:
        #     percentage = 0
        #     for obstacle in obstacles:
        #         percentage += n.get_intersect_percentage(obstacle)
        #         if percentage >= threshold:
        #             if n.value != 1:
        #                 changes.append(n)
        #                 n.value = 1
        #             break
        #     if percentage < threshold:
        #         if n.value:
        #             changes.append(n)
        #         n.value = 0
        # for change in changes:
        #     update_vertex(priority_queue, env, change)
        #     for n in change.neighbors:
        #         update_vertex(priority_queue, env, n)
        # compute_path(priority_queue, env)
        # new_path = show_path(env)
        # return new_path
        return self.solver.replan_path(obstacles)

    def detect(self, obstacles_list):
        obstacles = []
        for obstacle in obstacles_list:
            x1, x2, y1, y2 = obstacle.return_coordinate()
            closest_x = max(x1, min(self.pos[0], x2))
            closest_y = max(y1, min(self.pos[1], y2))
            distance = (closest_x - self.pos[0]) ** 2 + (closest_y - self.pos[1]) ** 2
            if distance <= self.r ** 2:
                obstacles.append(obstacle)
        return obstacles

    def nextPosition(self, goal):
        return goal
        # return PSO(30, 25, self.pos, goal)

    def decisionMaking(self, obstacles_list_before, obstacles_list_after, goal):
        self.decisionMaker.update(obstacles_list_before, obstacles_list_after, goal)
        return self.decisionMaker.decisionMaking(self)

        # self.onlyReplan.update(obstacles_list_before, obstacles_list_after, goal)
        # return self.onlyReplan.decisionMaking(self)

    def show_path(self):
        return self.solver.show_path()
