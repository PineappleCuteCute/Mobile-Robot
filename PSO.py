import numpy as np
from AABB import Obstacle
C = 1e6


def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def smooth(x1, y1, x2, y2, goal):
    goal_point_X, goal_point_Y = goal
    if x1 == x2:
        if y1 < y2:
            angle1 = -np.pi / 2
        else:
            angle1 = np.pi / 2
    else:
        angle1 = np.arctan((y1 - y2) / (x2 - x1))
    angle2 = np.arctan((y1 - goal_point_Y) / (goal_point_X - x1))
    return np.abs(angle1 - angle2)


def collide(pos, obs):
    for o in obs:
        if (np.abs(pos[0] - o.x) <= o.width / 2) or (np.abs(pos[1] - o.y) <= o .height / 2):
            return 1
    return 0


def optimization(last_pos, pos, goal, obs):
    last_pos_x, last_pos_y = last_pos
    pos_x, pos_y = pos
    goal_x, goal_y = goal
    w1, w2 = 0.2, 0.8
    return w1 * dist(pos_x, pos_y, goal_x, goal_y) + w2 * smooth(last_pos_x, last_pos_y, pos_x, pos_y, goal) \
           + C * collide(pos, obs)


def PSO(iteration, population, init, goal, low=-10, high=10, w=0.95, rp=0.6, rg=0.4, obstacles=None):
    # Initialize
    x_low, x_high = init[0] + low, init[0] + high
    y_low, y_high = init[1] + low, init[1] + high
    b_low = 0
    b_up = 1
    positions = []
    velocities = []
    p_best = []
    reduction = w / iteration
    for i in range(population):
        x = np.array([np.random.uniform(x_low, x_high), np.random.uniform(y_low, y_high)])
        positions.append(x)
        p_best.append(x)
        v = np.random.uniform(low=b_low-b_up, high=b_up-b_low, size=2)
        velocities.append(v)

    g_best = min(p_best, key=lambda arg: optimization(init, arg, goal, obstacles))

    # Update positions and velocities
    for _ in range(iteration):
        # w -= reduction
        for i in range(population):
            randp = np.random.uniform(0, 1, 2)
            randg = np.random.uniform(0, 1, 2)
            velocities[i] = velocities[i]*w + rp*randp*(p_best[i]-positions[i]) + rg*randg*(g_best-positions[i])
            positions[i] = positions[i] + velocities[i]
        for i in range(population):
            op = optimization(init, positions[i], goal, obstacles)
            if op < optimization(init, p_best[i], goal, obstacles):
                p_best[i] = positions[i]
                if op < optimization(init, g_best, goal, obstacles):
                    g_best = p_best[i]
    print(optimization(init, g_best, goal, obstacles))
    return g_best
