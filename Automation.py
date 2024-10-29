from QuadTree import main
# One of: [dense, maze, room, trap]
scenario = input("Enter scenario: ")
# One of: [Quad_Dstar_Tree, grid, Astar, OnlyReplan]
algorithm = input("Enter algorithm: ")
for i in range(1, 21):
    main(algorithm, scenario, scenario + str(i), False)
