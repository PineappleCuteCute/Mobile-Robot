import copy
import sys
import pygame
import time
import numpy as np
from pygame.locals import *
from sortedcontainers import SortedList
from Env import QuadTreeEnvironment, GridEnvironment
from robot import Robot
from PathManipulation import makeSpline, drawSpline, draw_path, draw_target, draw_start
from Colors import *
from Solver import DStarLiteSolver, AStarSolver
from DecisionMaking import FuzzyDecisionMaking, OnlyReplanDecision
from Obstacles import Obstacle, maps

# env_width = int(input("Enter width: "))
# env_height = int(input("Enter height: "))


def angle(p1, p2, p3):
    l = np.sqrt(((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) * ((p3[0] - p2[0]) ** 2 + (p3[1] - p2[1]) ** 2))
    a = max(((p1[0] - p2[0]) * (p3[0] - p2[0]) + (p1[1] - p2[1]) * (p3[1] - p2[1])) / l, -1)
    return np.pi - np.arccos(a)


# Get all modules of a given algorithm
def get_modules(algorithm):
    env_type = "grid" if algorithm == "grid" else "quadtree"
    planning_algo = "Astar" if algorithm == "Astar" else "DstarLite"
    decision_algo = "OnlyReplan" if algorithm == "OnlyReplan" else "Fuzzy"
    return env_type, planning_algo, decision_algo


def main(algorithm, scenario, test_map, interactive=True):
    env_width = env_height = 512
    # Pygame setup
    NORTH_PAD, SOUTH_PAD, LEFT_PAD, RIGHT_PAD = (int(env_height * 0.06), int(env_height * 0.16),
                                                 int(env_width * 0.06), int(env_width * 0.06))
    SCREEN_WIDTH = env_width + LEFT_PAD + RIGHT_PAD
    SCREEN_HEIGHT = env_height + NORTH_PAD + SOUTH_PAD
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
    pygame.display.set_caption("Quadtree Simulation")
    my_font = pygame.font.SysFont(None, SOUTH_PAD // 4)
    PATIENCE = 30

    # Modules
    env_type, planning_algo, decision_algo = get_modules(algorithm)
    print('Using:', env_type, planning_algo, decision_algo)

    # Initialization
    begin = (64, 500)
    end = (470, 180)
    mx, my, new_mx, new_my = None, None, None, None
    drawing = False
    done = False
    finished = False
    isStatic = True
    obstacles_list = []
    pause = False
    patience = 0
    env = None
    path = None
    past_path = []
    old_spl = None
    spl = None
    targets = 0
    collision = False
    start_time = None

    # Auto run
    if not interactive:
        if test_map in maps:
            obstacles_list = maps[test_map]
        done = True

    robot = Robot(begin, None, OnlyReplanDecision() if decision_algo == "OnlyReplan" else FuzzyDecisionMaking())

    while not finished:

        screen.fill(WHITE)
        if interactive:
            # Button
            button1 = pygame.draw.rect(screen, BLACK, (LEFT_PAD + int(env_width * 0.1), NORTH_PAD * 2 + env_height,
                                                       int(env_width * 0.2), int(SOUTH_PAD * 0.4)), 4)
            button1_text = my_font.render("Start", True, (0, 0, 0))
            button1_rect = button1_text.get_rect(center=button1.center)
            screen.blit(button1_text, button1_rect)

            button2 = pygame.draw.rect(screen, BLACK, (LEFT_PAD + int(env_width * 0.7), NORTH_PAD * 2 + env_height,
                                                       int(env_width * 0.2), int(SOUTH_PAD * 0.4)), 4)
            button2_text = my_font.render("Pause", True, (0, 0, 0))
            button2_rect = button2_text.get_rect(center=button2.center)
            screen.blit(button2_text, button2_rect)

            button3 = pygame.draw.rect(screen,
                                       BLACK,
                                       (LEFT_PAD + int(env_width * 0.4),
                                        NORTH_PAD * 2 + env_height,
                                        int(env_width * 0.2),
                                        int(SOUTH_PAD * 0.4)),
                                       4)
            button3_text = my_font.render("Static", True, (0, 0, 0))
            button3_rect = button3_text.get_rect(center=button3.center)
            screen.blit(button3_text, button3_rect)

            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = event.pos
                    if button1.collidepoint(mouse_x, mouse_y):
                        done = True
                        with open("Obstacles.py", 'a') as f:
                            f.write(",\n".join([o.__str__() for o in obstacles_list]))
                        if test_map in maps:
                            obstacles_list = maps[test_map]
                    elif button2.collidepoint(mouse_x, mouse_y):
                        pause = not pause
                    elif button3.collidepoint(mouse_x, mouse_y):
                        isStatic = not isStatic
                        if isStatic:
                            print("Static")
                        else:
                            print("Dynamic")
                    else:
                        mx, my = mouse_x, mouse_y
                        drawing = True
                        done = False
                if event.type == MOUSEBUTTONUP:
                    if drawing:
                        new_mx, new_my = event.pos
                        new_obstacle = Obstacle((mx+new_mx)/2,
                                                (my+new_my)/2,
                                                abs(new_mx-mx),
                                                abs(new_my-my),
                                                isStatic,
                                                np.random.randn(2) * 2)
                        obstacles_list.append(new_obstacle)
                        drawing = False
                if event.type == KEYDOWN:
                    if event.key == pygame.K_z and pygame.key.get_mods() & pygame.KMOD_CTRL:
                        if obstacles_list and not done:
                            obstacles_list.pop()

        if drawing:
            new_mx, new_my = pygame.mouse.get_pos()
            pygame.draw.rect(screen, BLACK, (min(mx, new_mx), min(my, new_my), abs(new_mx - mx), abs(new_my - my)))
        if not done:
            for obstacle in obstacles_list:
                obstacle.draw(screen)
            pygame.draw.rect(screen, BLACK, (LEFT_PAD, NORTH_PAD, env_width, env_height), 3)
        elif pause:
            continue
        else:
            if start_time is None:
                start_time = time.time()
            obstacles_list_before = copy.deepcopy(obstacles_list)
            for obstacle in obstacles_list:
                obstacle.move()
                obstacle.draw(screen)
            obstacles_list_after = obstacles_list

            if patience:
                robotX, robotY = robot.pos
                if (len(past_path) == 0) or robot.pos != past_path[-1]:
                    past_path.append(robot.pos)

                # Decision making
                decision_start = time.time()
                decision = robot.decisionMaking(obstacles_list_before, obstacles_list_after, local_goal)
                decision_end = time.time()
                with open('action/' + scenario + '/' + algorithm, 'a') as f:
                    f.write(f'{test_map} DECISION_MAKING {round(decision_end - decision_start, 4)}\n')

                for i in obstacles_list_before:
                    if -i.width <= 2 * (robotX - i.x) <= i.width and -i.height <= 2 * (robotY - i.y) <= i.height:
                        collision = True
                        print(f"{i.x} {i.width} {i.y} {i.height} {robotX} {robotY}")

                # print(decision)

                if decision == "Replan":
                    old_spl = copy.deepcopy(spl)
                    replan_start = time.time()
                    if planning_algo == 'Astar':
                        build_start = time.time()
                        # Choose env type
                        if env_type == 'quadtree':
                            env = QuadTreeEnvironment(LEFT_PAD + env_width / 2,
                                                      NORTH_PAD + env_height / 2,
                                                      env_width,
                                                      env_height)
                        elif env_type == 'grid':
                            env = GridEnvironment(LEFT_PAD + env_width / 2,
                                                  NORTH_PAD + env_height / 2,
                                                  env_width,
                                                  env_height)
                        env.update(obstacles_list)
                        env.build_env(robot.pos, end)
                        priority_queue = SortedList(key=lambda x: x.key)
                        env.goal.rhs = 0
                        env.goal.calculate_key()
                        priority_queue.add(env.goal)
                        robot.solver = AStarSolver(priority_queue, env)
                        build_end = time.time()
                        with open('action/' + scenario + '/' + algorithm, 'a') as f:
                            f.write(f'{test_map} ENV_DECOMPOSITION {round(build_end - build_start, 4)}\n')

                    path = robot.updatePath(obstacles_list)
                    spl = makeSpline(robot.pos, path, end)
                    local_goal = tuple(spl[:, 200])
                    targets = 1
                    replan_end = time.time()
                    with open('action/' + scenario + '/' + algorithm, 'a') as f:
                        f.write(f'{test_map} LOCAL_REPLAN {round(replan_end - replan_start, 4)}\n')
                if decision != "Stop":
                    robot.pos = robot.nextPosition(local_goal)

                if robot.reach(local_goal):
                    if local_goal == end:
                        finished = True
                    elif len(path) > 2 and robot.enter(path[1]):
                        path.pop(0)
                        env.current = path[0]
                    targets += 1
                    if targets * 200 < spl.shape[1]:
                        local_goal = tuple(spl[:, targets * 200])
                    else:
                        local_goal = end
                patience += 1
                if patience == PATIENCE:
                    patience = 0
            else:
                # Environment decomposition
                build_start = time.time()
                if env_type == 'quadtree':
                    env = QuadTreeEnvironment(LEFT_PAD + env_width / 2, NORTH_PAD + env_height / 2, env_width, env_height)
                elif env_type == 'grid':
                    env = GridEnvironment(LEFT_PAD + env_width / 2, NORTH_PAD + env_height / 2, env_width, env_height)
                env.update(obstacles_list)
                env.build_env(robot.pos, end)
                build_end = time.time()
                with open('action/' + scenario + '/' + algorithm, 'a') as f:
                    f.write(f'{test_map} ENV_DECOMPOSITION {round(build_end - build_start, 4)}\n')

                # Implementing path finding algorithm
                algo_start = time.time()
                priority_queue = SortedList(key=lambda x: x.key)
                env.goal.rhs = 0
                env.goal.calculate_key()
                priority_queue.add(env.goal)
                # Change between A star and D star
                if planning_algo == 'DstarLite':
                    robot.solver = DStarLiteSolver(priority_queue, env)
                elif planning_algo == 'Astar':
                    robot.solver = AStarSolver(priority_queue, env)
                path = robot.show_path()
                algo_end = time.time()
                with open('action/' + scenario + '/' + algorithm, 'a') as f:
                    f.write(f'{test_map} GLOBAL_PLANNING {round(algo_end - algo_start, 4)}\n')

                # Smoothen the path using Spline
                spl = makeSpline(robot.pos, path, end)

                targets = 1
                if targets * 200 < spl.shape[1]:
                    local_goal = tuple(spl[:, targets * 200])
                else:
                    local_goal = end
                patience += 1

            # Draw path
            drawSpline(old_spl, screen, DARK_GREY)
            draw_path(past_path, screen, GREEN)
            # draw_env_path(path, screen, robot.pos, end, draw_robot=True)
            drawSpline(spl, screen, YELLOW)
            # draw_local_goal(screen, local_goal)
            env.draw(screen, mode="boundary")
            draw_start(screen, begin)
            draw_target(screen, (end[0] - 10, end[1] - 64))
            robot.draw(screen)
            if robot.reach(end):
                finished = True
            time.sleep(0.1)
        pygame.display.update()

    end_time = time.time()


    with open("result/" + scenario + "/" + algorithm, "a") as f:
        d = 0
        for i in range(1, len(past_path)):
            d += np.sqrt(np.sum(np.square(np.array(past_path[i-1]) - np.array(past_path[i]))))
        count = smooth = 0
        for i in range(1, len(past_path) - 1):
            count += 1
            smooth += angle(past_path[i - 1], past_path[i], past_path[i + 1])
        # Write distance, smoothness and time of an execution to an output file
        # f.write(map + ' ' + str(round(d, 4)) + ' ' + str(round((smooth / count) * 180 / np.pi, 4)) +
        #         ' ' + str(round(end_time - start_time, 4)) + '\n')
        if collision:
            f.write(f"{test_map}: {round(d, 4)} {round((smooth / count) * 180 / np.pi, 4)} {round(end_time - start_time, 4)} Fail \n")
        else:
            f.write(f"{test_map}: {round(d, 4)} {round((smooth / count) * 180 / np.pi, 4)} {round(end_time - start_time, 4)} \n")
        # print(f"{start_time} {end_time}")


if __name__ == '__main__':
    # One of: [dense, maze, room, trap]
    s = 'real' #input('Enter scenario: ')
    # One of: [Quad_Dstar_Tree, grid, Astar, OnlyReplan]
    algo = 'Quad_Dstar_Tree' #input('Enter algorithm: ')
    # From 1 to 20
    input_map = '7' #input('Enter map number: ')
    main(algo, s, s + input_map)
