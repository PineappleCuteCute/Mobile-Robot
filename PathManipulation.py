from scipy import interpolate
import numpy as np
import pygame
from Colors import *

# def make_spline(x_ys):
#     # x_ys = np.array([[0, 1, 1, 2, 3, 4, 5, 7], [1, 2, 3, 5, 5.5, 7, 7, 8.5]])
#     x_y_new = x_ys - x_ys[:, :1]
#     rotation = np.array([[x_y_new[0][-1], x_y_new[1][-1]], [-x_y_new[1][-1], x_y_new[0][-1]]]).astype(np.float64)
#     rotation /= np.sqrt(x_y_new[0][-1] ** 2 + x_y_new[1][-1] ** 2)
#     x_y_transformed = np.matmul(rotation, x_y_new)
#
#     tck = interpolate.splrep(x_y_transformed[0], x_y_transformed[1], s=1)
#     x_new = np.linspace(0, x_y_transformed[0][-1], 1000)
#     y_new = interpolate.splev(x_new, tck)
#
#     re_rotation = rotation * np.array([[1, -1], [-1, 1]])
#     spl = np.matmul(re_rotation, np.array([x_new, y_new])) + x_ys[:, :1]
#
#     return spl


def makeSpline(robotPos, path, goal):
    try:
        x, y = np.array([p.x for p in path]), np.array([p.y for p in path])
        x[0] = robotPos[0]
        y[0] = robotPos[1]
        x[-1] = goal[0]
        y[-1] = goal[1]
        n = len(path)
        i = np.arange(n)
        interp_i = np.linspace(0, n - 1, n * 1000)

        xs = interpolate.interp1d(i, x, kind='cubic', assume_sorted=True)(interp_i)
        ys = interpolate.interp1d(i, y, kind='cubic', assume_sorted=True)(interp_i)

        # if window:
        #     for k in range(n * 1000 - 1):
        #         pygame.draw.line(window, (0, 255, 0), (xs[k], ys[k]), (xs[k+1], ys[k+1]), 3)
        return np.array([xs, ys])
    except:
        return np.array([np.linspace(robotPos[0], goal[0], 1000), np.linspace(robotPos[1], goal[1], 1000)])


def drawSpline(spl, window, color=GREEN):
    if spl is None:
        return
    for i in range(spl.shape[1] - 1):
        pygame.draw.line(window, color, (spl[0][i], spl[1][i]), (spl[0][i+1], spl[1][i+1]), 3)


def draw_env_path(path, window, start, end, draw_robot=True, color=GREEN):
    n = len(path)
    if n:
        # path = np.array([[p.x for p in path], [p.y for p in path]])
        if type(path[-1]) == tuple:
            pygame.draw.line(window, color, (path[0].x, path[0].y), path[-1], 3)
        else:
            for i in range(n - 1):
                start_pos = (path[i].x, path[i].y) if i else start
                end_pos = (path[i + 1].x, path[i + 1].y) if i < n - 2 else end
                pygame.draw.line(window, color, start_pos, end_pos, 3)
        if draw_robot:
            path[0].draw(window)
            if n > 1 and type(path[1]) != tuple:
                path[1].draw(window)


def draw_path(path, window, color):
    n = len(path)
    for i in range(n - 1):
        start_pos = (path[i][0], path[i][1])
        end_pos = (path[i + 1][0], path[i + 1][1])
        pygame.draw.line(window, color, start_pos, end_pos, 3)


def draw_target(window, target):
    target_img = pygame.image.load('flag.png')
    window.blit(target_img, target)

def draw_start(window, start):
    pygame.draw.circle(window, GREEN, start, 6, 0)

def draw_local_goal(window, local_goal):
    pygame.draw.circle(window, (255, 0, 0), local_goal, 2.5)
