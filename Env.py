from abc import ABC, abstractmethod
import numpy as np
from AABB import Node
from itertools import chain


def my_iter(iterable):
    if type(iterable[0]) == list:
        return chain.from_iterable(iterable)
    return iterable


class Environment(ABC):
    def __init__(self, x, y, env_width, env_height):
        self.root = Node(x, y, env_width, env_height)
        self.nodes = []
        self.start = None
        self.goal = None
        self.current = None

    @abstractmethod
    def update(self, obstacles):
        pass

    @abstractmethod
    def build_env(self, start, goal):
        pass

    def draw(self, window, mode="full"):
        if mode == 'full':
            # self.current.draw(window)
            for node in my_iter(self.nodes):
                # if node == self.start:
                #     start_text = my_font.render("Current", True, (0, 0, 0))
                #     start_rect = start_text.get_rect(center=tuple([node.x, node.y]))
                #     window.blit(start_text, start_rect)
                # if node == self.goal:
                #     goal_text = my_font.render("Goal", True, (0, 0, 0))
                #     goal_rect = goal_text.get_rect(center=tuple([node.x, node.y]))
                #     window.blit(goal_text, goal_rect)
                node.draw(window)
                # for neighbor in node.neighbors:
                #    pygame.draw.line(window, GREEN, (node.x, node.y), (neighbor.x, neighbor.y))
        if mode == 'boundary':
            self.root.draw(window)
        if mode == 'none':
            return


class QuadTreeEnvironment(Environment):
    def __init__(self, x, y, env_width, env_height):
        super().__init__(x, y, env_width, env_height)
        self.nodes = [self.root]

    def update(self, obstacles):
        self.root.update_percentage_and_split(obstacles)

    def build_env(self, start, goal):
        nodes = []
        self.add_start(start)
        self.add_goal(goal)
        for leaf in self.root.get_leaves():
            leaf.update_neighbors()
            nodes.append(leaf)
            leaf.h = np.sqrt(np.square(leaf.x - self.goal.x) + np.square(leaf.y - self.goal.y))
        self.nodes = nodes
        self.current = self.start

    def add_start(self, start):
        current = self.root
        while current.get_children():
            current = current.get_quadrant(start)
        self.start = current

    def add_goal(self, goal):
        current = self.root
        while current.get_children():
            current = current.get_quadrant(goal)
        self.goal = current

    # def clear(self):
    #     self.nodes = [self.root]
    #     self.start = None
    #     self.goal = None
    #     self.current = None
    # 
    # def solve(self):
    #     priority_queue = PriorityQueue()
    #     self.goal.rhs = 0
    #     priority_queue.insert(self.goal)
    #     compute_path(priority_queue, self)
    # 
    # def show_path(self):
    #     return show_path(self)


class GridEnvironment(Environment):
    def __init__(self, x, y, env_width, env_height, size=32):
        super().__init__(x, y, env_width, env_height)
        self.size = size
        self.cell_width = env_width / size
        self.cell_height = env_height / size
        self.left_pad = x - env_width / 2
        self.north_pad = y - env_height / 2
        self.nodes = [[Node(x + (2 * j - size + 1) * env_width / (2 * size),
                            y + (2 * i - size + 1) * env_height / (2 * size),
                            self.cell_width,
                            self.cell_height) for j in range(size)] for i in range(size)]

    def update(self, obstacles):
        for obstacle in obstacles:
            x1, x2, y1, y2 = obstacle.return_coordinate()
            lb_x, ub_x = int((x1 - self.left_pad) / self.cell_width), int(np.ceil((x2 - self.left_pad) / self.cell_width))
            lb_y, ub_y = int((y1 - self.north_pad) / self.cell_height), int(np.ceil((y2 - self.north_pad) / self.cell_height))
            for r in range(lb_y, ub_y):
                for c in range(lb_x, ub_x):
                    self.nodes[r][c].value = 1

    def build_env(self, start, goal):
        start_x, start_y = int((start[0] - self.left_pad) / self.cell_width), int(
            (start[1] - self.north_pad) / self.cell_height)
        self.start = self.nodes[start_y][start_x]

        end_x, end_y = int((goal[0] - self.left_pad) / self.cell_width), int(
            (goal[1] - self.north_pad) / self.cell_height)
        self.goal = self.nodes[end_y][end_x]

        self.current = self.start

        for i in range(self.size):
            for j in range(self.size):
                node = self.nodes[i][j]
                node.h = np.sqrt(np.square(node.x - self.goal.x) + np.square(node.y - self.goal.y))
                if node.value == -1:
                    node.value = 0
                for dx, dy in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                    n_x, n_y = i + dx, j + dy
                    if 0 <= n_x < self.size and 0 <= n_y < self.size:
                        node.neighbors.append(self.nodes[n_x][n_y])
