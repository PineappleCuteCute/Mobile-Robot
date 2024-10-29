import pygame
from pygame.locals import *
import numpy as np
import sys

running = True
size = int(input("Enter size: "))
GRID_SIZE = 30
GRAPH_SIZE = GRID_SIZE * size
SCREEN_WIDTH = 2 * GRAPH_SIZE + 150
SCREEN_HEIGHT = GRAPH_SIZE + 150
MIDDLE = SCREEN_WIDTH // 2
top_lefts = [(50, 100), (100 + GRAPH_SIZE, 100)]
mode = ""
MODE = ["start_goal", "obstacles", "unknown obstacles"]

alpha = 0.1

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
pygame.display.set_caption("Mobile robot simulation")


class PriorityQueue:
    def __init__(self):
        self.queue = []

    def is_empty(self):
        return len(self.queue) == 0

    def pop(self):
        v = self.queue.pop(0)
        v.in_queue = False
        return v

    def remove(self, vertex):
        self.queue.remove(vertex)
        vertex.in_queue = False

    def insert(self, vertex):
        key = vertex.calculate_key()
        keys = [v.calculate_key() for v in self.queue]
        insert_idx = binary_search(key, keys, 0, len(self.queue))
        self.queue.insert(insert_idx, vertex)
        vertex.in_queue = True

    def top_key(self):
        return self.queue[0].calculate_key()


class Graph:
    def __init__(self, size):
        self.size = size
        self.graph = [[None for _ in range(size)] for _ in range(size)]
        self.grid = np.ones((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                self.graph[i][j] = Vertex(i, j, np.inf, np.inf)
        self.start = None
        self.current = None
        self.goal = None

    def update_start(self, start, update_current=False):
        self.start = self.graph[start[0]][start[1]]
        if update_current:
            self.current = self.start

    def update_goal(self, goal):
        self.goal = self.graph[goal[0]][goal[1]]
        self.goal.rhs = 0

    def succ(self, vertex):
        answer = []
        x, y = vertex.coordinate
        x_low, x_high, y_low, y_high = x - 1, x + 1, y - 1, y + 1
        if x_low == -1:
            x_low = 0
        if x_high == self.size:
            x_high = self.size - 1
        if y_low == -1:
            y_low = 0
        if y_high == self.size:
            y_high = self.size - 1
        for i in range(x_low, x_high + 1):
            for j in range(y_low, y_high + 1):
                if (i, j) == (x, y):
                    continue
                answer.append(self.graph[i][j])
        return answer

    def pred(self, vertex):
        answer = []
        x, y = vertex.coordinate
        x_low, x_high, y_low, y_high = x - 1, x + 1, y - 1, y + 1
        if x_low == -1:
            x_low = 0
        if x_high == self.size:
            x_high = self.size - 1
        if y_low == -1:
            y_low = 0
        if y_high == self.size:
            y_high = self.size - 1
        for i in range(x_low, x_high + 1):
            for j in range(y_low, y_high + 1):
                if (i, j) == (x, y):
                    continue
                answer.append(self.graph[i][j])
        return answer

    def distance(self, u, v):
        return np.sqrt((u.x - v.x) ** 2 + (u.y - v.y) ** 2)

    def cost(self, u, v):
        if np.abs(u.x - v.x) > 1 or np.abs(u.y - v.y) > 1:
            print('Error')
        else:
            u_x, u_y = u.coordinate
            v_x, v_y = v.coordinate
            return self.distance(u, v)/((self.grid[u_x][u_y] + 1e-6) * (self.grid[v_x][v_y] + 1e-6))

    def update_grid(self, coordinate, p):
        i, j = coordinate
        self.grid[i][j] = p

    def calculate_rhs(self, vertex):
        ans = np.inf
        for succ in self.succ(vertex):
            tmp = succ.g + self.cost(succ, vertex)
            if tmp < ans:
                ans = tmp
        vertex.rhs = ans
        return ans


class Vertex:
    def __init__(self, x, y, g, rhs):
        self.coordinate = (x, y)
        self.x = x
        self.y = y
        self.g = g
        self.rhs = rhs
        self.h = 0
        self.in_queue = False

    def calculate_key(self):
        return [min(self.g, self.rhs) + self.h, min(self.g, self.rhs)]


def compute_path(queue, graph):
    for i in range(graph.size):
        for j in range(graph.size):
            graph.graph[i][j].h = (1 - alpha) * graph.graph[i][j].h + alpha * graph.distance(graph.graph[i][j], graph.current)
    while (compare_array(queue.top_key(), graph.current.calculate_key())) or \
            (graph.current.g != graph.current.rhs):
        v = queue.pop()
        if v.g > v.rhs:
            v.g = v.rhs
            for u in graph.succ(v):
                update_vertex(queue, graph, u)
        else:
            v.g = np.inf
            update_vertex(queue, graph, v)
            for u in graph.succ(v):
                update_vertex(queue, graph, u)
        if queue.is_empty():
            break


def update_vertex(queue, graph, vertex):
    if vertex != graph.goal:
        graph.calculate_rhs(vertex)
    if vertex.in_queue:
        queue.remove(vertex)
    if vertex.g != vertex.rhs:
        queue.insert(vertex)


def binary_search(key, keys, start, end):
    if start == end:
        return start
    else:
        middle = (end + start) // 2
        if key[0] == keys[middle][0]:
            if key[1] == keys[middle][1]:
                return middle
            elif key[1] < keys[middle][1]:
                return binary_search(key, keys, start, middle)
            else:
                return binary_search(key, keys, middle + 1, end)
        elif key < keys[middle]:
            return binary_search(key, keys, start, middle)
        else:
            return binary_search(key, keys, middle + 1, end)


def compare_array(a, b):
    for i in range(max(len(a), len(b))):
        if i > len(b):
            return False
        if a[i] < b[i]:
            return True
        elif a[i] > b[i]:
            return False


def show_path(graph):
    path = [graph.current.coordinate]
    current = graph.current
    while current != graph.goal:
        cur_succ = graph.succ(current)
        min_idx = np.argmin([v.g for v in cur_succ])
        current = cur_succ[min_idx]
        path.append(current.coordinate)
    return path


def draw_graph(graph, top_left):
    my_font = pygame.font.SysFont(None, int(GRID_SIZE * 0.7))
    for i in range(graph.size):
        for j in range(graph.size):
            if graph.graph[i][j] == graph.start:
                rect = pygame.draw.rect(screen, (0, 0, 255),
                                        (top_left[0] + GRID_SIZE * i, top_left[1] + GRID_SIZE * j, GRID_SIZE, GRID_SIZE))
                start_text = my_font.render("Start", True, (0, 0, 0))
                start_rect = start_text.get_rect(center=rect.center)
                screen.blit(start_text, start_rect)
            elif graph.graph[i][j] == graph.goal:
                rect = pygame.draw.rect(screen, (255, 0, 0),
                                        (top_left[0] + GRID_SIZE * i, top_left[1] + GRID_SIZE * j, GRID_SIZE, GRID_SIZE))
                goal_text = my_font.render("Goal", True, (0, 0, 0))
                goal_rect = goal_text.get_rect(center=rect.center)
                screen.blit(goal_text, goal_rect)
            else:
                pygame.draw.rect(screen, (0, 0, 0), (top_left[0] + GRID_SIZE * i, top_left[1] + GRID_SIZE * j,
                                                     GRID_SIZE, GRID_SIZE), graph.grid[i][j])


def show_title(coordinate, title, fontsize=20):
    font = pygame.font.SysFont(None, fontsize)
    title = font.render(title, True, (0, 0, 0))
    title_rect = title.get_rect(center=coordinate)
    screen.blit(title, title_rect)


def draw_path(path, top_left):
    pygame.draw.circle(screen, (0, 255, 0), (top_left[0] + (path[0][0] + 0.5) * GRID_SIZE,
                                             top_left[1] + (path[0][1] + 0.5) * GRID_SIZE), 5)
    for i in range(len(path) - 1):
        start_pos = (top_left[0] + (path[i][0] + 0.5) * GRID_SIZE, top_left[1] + (path[i][1] + 0.5) * GRID_SIZE)
        end_pos = (top_left[0] + (path[i+1][0] + 0.5) * GRID_SIZE, top_left[1] + (path[i+1][1] + 0.5) * GRID_SIZE)
        pygame.draw.line(screen, (0, 255, 0), start_pos, end_pos, 3)


real = Graph(size)
graph = Graph(size)
priority_queue = PriorityQueue()
while running:
    screen.fill((255, 255, 255))

    font = pygame.font.SysFont(None, 20)
    start_goal_rect = pygame.draw.rect(screen, (0, 0, 0), (MIDDLE - 450, 30, 150, 40), 4)
    start_goal_text = font.render("Goal / Start", True, (0, 0, 0))
    start_goal_text_rect = start_goal_text.get_rect(center=start_goal_rect.center)
    screen.blit(start_goal_text, start_goal_text_rect)

    obstacles_rect = pygame.draw.rect(screen, (0, 0, 0), (MIDDLE - 200, 30, 150, 40), 4)
    obstacles_text = font.render("Obstacles", True, (0, 0, 0))
    obstacles_text_rect = obstacles_text.get_rect(center=obstacles_rect.center)
    screen.blit(obstacles_text, obstacles_text_rect)

    unknown_obstacles_rect = pygame.draw.rect(screen, (0, 0, 0), (MIDDLE + 50, 30, 150, 40), 4)
    unknown_obstacles_text = font.render("Unknown Obstacles", True, (0, 0, 0))
    unknown_obstacles_text_rect = unknown_obstacles_text.get_rect(center=unknown_obstacles_rect.center)
    screen.blit(unknown_obstacles_text, unknown_obstacles_text_rect)

    running_rect = pygame.draw.rect(screen, (0, 0, 0), (MIDDLE + 300, 30, 150, 40), 4)
    running_text = font.render("Running", True, (0, 0, 0))
    running_text_rect = running_text.get_rect(center=running_rect.center)
    screen.blit(running_text, running_text_rect)

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == MOUSEBUTTONDOWN:
            mx, my = event.pos
            if start_goal_rect.collidepoint(mx, my):
                mode = "start_goal"
            elif obstacles_rect.collidepoint(mx, my):
                mode = "obstacles"
            elif unknown_obstacles_rect.collidepoint(mx, my):
                mode = "unknown obstacles"
            elif running_rect.collidepoint(mx, my):
                mode = "running"
                compute_path(priority_queue, graph)
            else:
                if mode in MODE:
                    i = int((mx - top_lefts[0][0]) / GRID_SIZE)
                    j = int((my - top_lefts[0][1]) / GRID_SIZE)
                    if mode == "start_goal":
                        if event.button == 1:
                            real.update_start((i, j))
                            graph.update_start((i, j), update_current=True)
                        if event.button == 3:
                            real.update_goal((i, j))
                            graph.update_goal((i, j))
                            priority_queue.insert(graph.goal)
                    if mode == "obstacles":
                        real.update_grid((i, j), 0)
                        graph.update_grid((i, j), 0)
                    if mode == "unknown obstacles":
                        real.update_grid((i, j), 0)
                else:
                    print("Not a valid mode")
                    running = False

    draw_graph(real, top_lefts[0])
    draw_graph(graph, top_lefts[1])
    show_title((top_lefts[0][0] + GRAPH_SIZE / 2, top_lefts[0][1] + GRAPH_SIZE + 20), "Real environment")
    show_title((top_lefts[1][0] + GRAPH_SIZE / 2, top_lefts[1][1] + GRAPH_SIZE + 20), "Robot's environment")

    if mode == "running":
        changes = []
        for v in graph.succ(graph.current):
            i, j = v.coordinate
            if graph.grid[i][j] != real.grid[i][j]:
                changes.append((i, j))
        for change in changes:
            i, j = change
            graph.update_grid((i, j), real.grid[i][j])
            update_vertex(priority_queue, graph, graph.graph[i][j])
            for pred in graph.pred(graph.graph[i][j]):
                update_vertex(priority_queue, graph, pred)
        # if changes:
        #     for vertex in graph.succ(graph.current):
        #         x, y = vertex.coordinate
        #         if graph.grid[x][y] == 0:
        #             changes = []
        #             for v in graph.succ(vertex):
        #                 i, j = v.coordinate
        #                 if graph.grid[i][j] != real.grid[i][j]:
        #                     changes.append((i, j))
        #             for change in changes:
        #                 i, j = change
        #                 graph.update_grid((i, j), real.grid[i][j])
        #                 update_vertex(priority_queue, graph, graph.graph[i][j])
        #                 for pred in graph.pred(graph.graph[i][j]):
        #                     update_vertex(priority_queue, graph, pred)
            compute_path(priority_queue, graph)

        draw_graph(real, top_lefts[0])
        draw_graph(graph, top_lefts[1])
        path = show_path(graph)
        draw_path(path, top_lefts[1])

        if graph.current != graph.goal:
            graph.current = graph.graph[path[1][0]][path[1][1]]
        else:
            running = False

    pygame.display.update()
    pygame.time.wait(1000)