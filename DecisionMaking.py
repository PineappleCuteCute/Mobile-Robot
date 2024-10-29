from abc import ABC, abstractmethod
import numpy as np


def angle(x1, y1, x2, y2):
    return np.arccos((x1 * x2 + y1 * y2) / (np.sqrt(x1 * x1 + y1 * y1) * np.sqrt(x2 * x2 + y2 * y2) + 1e-6))


class DecisionMaking(ABC):

    def __init__(self):
        self.goal = None
        self.obstacles_list_before = None
        self.obstacles_list_after = None

    def update(self, obstacles_list_before, obstacles_list_after, goal):
        self.obstacles_list_before = obstacles_list_before
        self.obstacles_list_after = obstacles_list_after
        self.goal = goal

    @abstractmethod
    def decisionMaking(self, rb):
        pass


class OnlyReplanDecision(DecisionMaking):
    def decisionMaking(self, rb):
        decision = "No"
        for i in range(len(self.obstacles_list_before)):
            x1 = self.obstacles_list_before[i].x
            y1 = self.obstacles_list_before[i].y

            rb_next = rb.nextPosition(self.goal)

            distance = np.sqrt((rb.pos[0] - x1)*(rb.pos[0] - x1) + (rb.pos[1] - y1)*(rb.pos[1] - y1))
            phi = angle(rb_next[0] - rb.pos[0], rb_next[1] - rb.pos[1], x1 - rb.pos[0], y1 - rb.pos[1])
            if distance < rb.r and phi <= np.pi / 2:
                decision = "Replan"
        return decision

class FuzzyDecisionMaking(DecisionMaking):
    def __init__(self):
        super().__init__()
        self.ANGLE_F = 25
        self.ANGLE_DF = 35
        self.ANGLE_DS = 55
        self.ANGLE_S = 65

        self.DELTAD_A = 2.5
        self.DELTAD_UA = 1.5
        self.DELTAD_UC = -1.5
        self.DELTAD_C = -2.5

        self.DELTAPHI_A = 10
        self.DELTAPHI_LAA = 7
        self.DELTAPHI_LAU = 5
        self.DELTAPHI_ULA = 3
        self.DELTAPHI_ULC = -3
        self.DELTAPHI_LCU = -5
        self.DELTAPHI_LCC = -7
        self.DELTAPHI_C = -10

    def convertphi(self, phi):
        aS, aD, aF = 0, 0, 0
        if phi > self.ANGLE_S:
            aS = 1
        elif self.ANGLE_S >= phi >= self.ANGLE_DS:
            aS = (self.ANGLE_S - phi) / (self.ANGLE_S - self.ANGLE_DS)
            aD = 1 - (phi - self.ANGLE_DS) / (self.ANGLE_S - self.ANGLE_DS)
        elif self.ANGLE_DS > phi > self.ANGLE_DF:
            aD = 1
        elif self.ANGLE_DF >= phi >= self.ANGLE_F:
            aD = (self.ANGLE_DF - phi) / (self.ANGLE_DF - self.ANGLE_F)
            aF = 1 - (phi - self.ANGLE_F) / (self.ANGLE_DF - self.ANGLE_F)
        elif self.ANGLE_F > phi:
            aF = 1
        m = max(aS, aD, aF)
        if m == aS:
            return "S"
        elif m == aD:
            return "D"
        else:
            return "F"

    def convertdeltad(self, deltad):
        ddA, ddU, ddC = 0, 0, 0
        if deltad > self.DELTAD_A:
            ddA = 1
        elif self.DELTAD_A >= deltad >= self.DELTAD_UA:
            ddA = (self.DELTAD_A - deltad) / (self.DELTAD_A - self.DELTAD_UA)
            ddU = (deltad - self.DELTAD_UA) / (self.DELTAD_A - self.DELTAD_UA)
        elif self.DELTAD_UA > deltad > self.DELTAD_UC:
            ddU = 1
        elif self.DELTAD_UC > deltad > self.DELTAD_C:
            ddU = (self.DELTAD_UC - deltad) / (self.DELTAD_UC - self.DELTAD_C)
            ddC = (deltad - self.DELTAD_C) / (self.DELTAD_UC - self.DELTAD_C)
        elif self.DELTAD_C > deltad:
            ddC = 1
        m = max(ddA, ddU, ddC)
        if m == ddA:
            return "A"
        elif m == ddU:
            return "U"
        else:
            return "C"

    def convertdeltaphi(self, deltaphi):
        dpA, dpLA, dpU, dpLC, dpC = 0, 0, 0, 0, 0
        if deltaphi > self.DELTAPHI_A:
            dpA = 1
        elif self.DELTAPHI_A >= deltaphi >= self.DELTAPHI_LAA:
            dpA = (self.DELTAPHI_A - deltaphi) / (self.DELTAPHI_A - self.DELTAPHI_LAA)
            dpLA = (deltaphi - self.DELTAPHI_LAA) / (self.DELTAPHI_A - self.DELTAPHI_LAA)
        elif self.DELTAPHI_LAA > deltaphi > self.DELTAPHI_LAU:
            dpLA = 1
        elif self.DELTAPHI_LAU >= deltaphi >= self.DELTAPHI_ULA:
            dpLA = (self.DELTAPHI_LAU - deltaphi) / (self.DELTAPHI_LAU - self.DELTAPHI_ULA)
            dpU = (deltaphi - self.DELTAPHI_ULA) / (self.DELTAPHI_LAU - self.DELTAPHI_ULA)
        elif self.DELTAPHI_ULA > deltaphi > self.DELTAPHI_ULC:
            dpU = 1
        elif self.DELTAPHI_ULC >= deltaphi >= self.DELTAPHI_LCU:
            dpU = (self.DELTAPHI_ULC - deltaphi) / (self.DELTAPHI_ULC - self.DELTAPHI_LCU)
            dpLC = (deltaphi - self.DELTAPHI_LCU) / (self.DELTAPHI_ULC - self.DELTAPHI_LCU)
        elif self.DELTAPHI_LCU > deltaphi > self.DELTAPHI_LCC:
            dpLC = 1
        elif self.DELTAPHI_LCC >= deltaphi >= self.DELTAPHI_C:
            dpLC = (self.DELTAPHI_LCC - deltaphi) / (self.DELTAPHI_LCC - self.DELTAPHI_C)
            dpC = (deltaphi - self.DELTAPHI_C) / (self.DELTAPHI_LCC - self.DELTAPHI_C)
        elif self.DELTAPHI_C > deltaphi:
            dpC = 1
        m = max(dpA, dpLA, dpU, dpLC, dpC)
        if m == dpA:
            return "A"
        elif m == dpLA:
            return "LA"
        elif m == dpU:
            return "U"
        elif m == dpLC:
            return "LC"
        else:
            return "C"

    def truthtable(self, phi, deltad, deltaphi):
        if deltad == "A":
            if phi == "F":
                if deltaphi == "C" or deltaphi == "LC" or deltaphi == "U":
                    return "Stop"
                elif deltaphi == "LA" or deltaphi == "A":
                    return "No"
            else:
                return "No"
        elif deltad == "U":
            if phi == "S":
                return "No"
            elif phi == "D":
                if deltaphi == "C" or deltaphi == "LC":
                    return "Replan"
                elif deltaphi == "U" or deltaphi == "LA" or deltaphi == "A":
                    return "No"
            elif phi == "F":
                if deltaphi == "C" or deltaphi == "LC" or deltaphi == "U":
                    return "Stop"
                elif deltaphi == "LA" or deltaphi == "A":
                    return "No"
        elif deltad == "C":
            if phi == "S":
                if deltaphi == "LC":
                    return "Stop"
                elif deltaphi == "U":
                    return "Replan"
                elif deltaphi == "C" or deltaphi == "LA" or deltaphi == "A":
                    return "No"
            elif phi == "D":
                if deltaphi == "C" or deltaphi == "LC":
                    return "Stop"
                elif deltaphi == "U":
                    return "Replan"
                elif deltaphi == "LA" or deltaphi == "A":
                    return "No"
            elif phi == "F":
                if deltaphi == "C" or deltaphi == "LC" or deltaphi == "U":
                    return "Replan"
                elif deltaphi == "LA" or deltaphi == "A":
                    return "Stop"
        return "No"

    def fuzzyDecisionMaking(self, phit, phit_next, dt, dt_next):
        phi = self.convertphi(phit / np.pi * 180)
        deltaphi = self.convertdeltaphi((phit_next - phit) / np.pi * 180)
        deltad = self.convertdeltad((dt_next - dt))
        return self.truthtable(phi, deltad, deltaphi)

    def decisionMaking(self, rb):
        decision = "No"
        for i in range(len(self.obstacles_list_before)):
            x1 = self.obstacles_list_before[i].x
            y1 = self.obstacles_list_before[i].y
            x2 = self.obstacles_list_after[i].x
            y2 = self.obstacles_list_after[i].y
            if x1 == x2 and y1 == y2: continue
            x1, y1 = min(self.obstacles_list_before[i].get_corners(),
                         key=lambda x: (rb.pos[0] - x[0]) ** 2 + (rb.pos[1] - x[1]) ** 2)
            x2, y2 = min(self.obstacles_list_after[i].get_corners(),
                         key=lambda x: (rb.pos[0] - x[0]) ** 2 + (rb.pos[1] - x[1]) ** 2)
            distance = np.sqrt((rb.pos[0] - x1) * (rb.pos[0] - x1) + (rb.pos[1] - y1) * (rb.pos[1] - y1))
            if distance < rb.r:
                distance_next = np.sqrt((rb.pos[0] - x2) * (rb.pos[0] - x2) + (rb.pos[1] - y2) * (rb.pos[1] - y2))
                rb_next = rb.nextPosition(self.goal)
                phi = angle(rb_next[0] - rb.pos[0], rb_next[1] - rb.pos[1], x1 - rb.pos[0], y1 - rb.pos[1])
                phi_next = angle(rb_next[0] - rb.pos[0], rb_next[1] - rb.pos[1], x2 - rb.pos[0], y2 - rb.pos[1])
                decision_temp = self.fuzzyDecisionMaking(phi, phi_next, distance, distance_next)
                if decision_temp == "Replan":
                    return decision_temp
                elif decision_temp == "Stop":
                    decision = decision_temp
        return decision