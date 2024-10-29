from abc import ABC, abstractmethod
import numpy as np
from AABB import cost


class PriorityQueueSolver(ABC):
    def __init__(self, queue=None, graph=None):
        self.queue = queue
        self.graph = graph

    @abstractmethod
    def compute_path(self):
        pass

    @abstractmethod
    def update_vertex(self, vertex):
        pass

    @abstractmethod
    def replan_path(self, obstacles):
        pass

    @abstractmethod
    def show_path(self):
        pass


class DStarLiteSolver(PriorityQueueSolver):
    def compute_path(self):
        while (self.queue and (self.queue[0].key < self.graph.current.calculate_key())) or \
                (self.graph.current.g != self.graph.current.rhs):
            v = self.queue.pop(0)
            if v.g > v.rhs:
                v.g = v.rhs
                for u in v.neighbors:
                    self.update_vertex(u)
            else:
                v.g = np.inf
                self.update_vertex(v)
                for u in v.neighbors:
                    self.update_vertex(u)

    def show_path(self):
        self.compute_path()
        path = [self.graph.current]
        current = self.graph.current
        while current != self.graph.goal:
            current = min(current.neighbors, key=lambda x: x.g)
            path.append(current)
        return path

    def update_vertex(self, vertex):
        if vertex != self.graph.goal:
            vertex.calculate_rhs()
        self.queue.discard(vertex)
        if vertex.g != vertex.rhs:
            vertex.calculate_key()
            self.queue.add(vertex)

    def replan_path(self, obstacles):
        threshold = 1e-3
        changes = []
        for n in self.graph.current.neighbors:
            percentage = 0
            for obstacle in obstacles:
                percentage += n.get_intersect_percentage(obstacle)
                if percentage >= threshold:
                    if n.value != 1:
                        changes.append(n)
                        n.value = 1
                    break
            if percentage < threshold:
                if n.value:
                    changes.append(n)
                n.value = 0
        for change in changes:
            self.update_vertex(change)
            for n in change.neighbors:
                self.update_vertex(n)
        new_path = self.show_path()
        return new_path


class AStarSolver(PriorityQueueSolver):
    def compute_path(self):
        visited = set()
        while self.queue:
            v = self.queue.pop(0)
            if v == self.graph.current:
                break
            visited.add((v.x, v.y))
            for u in v.neighbors:
                if (u.x, u.y) not in visited:
                    self.queue.discard(u)
                    u.rhs = min(u.rhs, v.rhs + cost(u, v))
                    u.calculate_key()
                    self.queue.add(u)

    def update_vertex(self, vertex):
        pass

    def show_path(self):
        self.compute_path()
        path = [self.graph.current]
        current = self.graph.current
        while current != self.graph.goal:
            current = min(current.neighbors, key=lambda x: x.rhs)
            path.append(current)
        return path

    def replan_path(self, obstacles):
        return self.show_path()

