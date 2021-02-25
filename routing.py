import sys
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import Model, GRB, quicksum


# Capacitated Vehicle Routing Problem
def main():


    ########################## Prepare data ##########################


    # determine the number of clients, vehicle capacity, seed
    if len(sys.argv) == 1:
        argv = [10,10,1]
    else:
        argv = sys.argv[1:]

    # seed to be used for randomness
    rnd = np.random
    rnd.seed(int(argv[2]))

    # number of clients
    n = int(argv[0])

    # create the clients and factory nodes
    x_coord = rnd.rand(n+1)*200
    y_coord = rnd.rand(n+1)*100

    # plot the nodes before the route optimization
    plt.plot(x_coord[0], y_coord[0], c='r', marker='s')
    plt.scatter(x_coord[1:], y_coord[1:], c='b')
    plt.show()


    ########################## Optimize ##########################


    N = [i for i in range(1,n+1)] # clients
    V = [0] + N # factory + clients

    A = [(i,j) for i in V for j in V if i!=j] # arcs

    D = {(i,j): np.hypot(x_coord[i] - x_coord[j], y_coord[i] - y_coord[j]) for i,j in A} # distances between nodes

    T = {(i,j): rnd.randint(1,10) for i,j in A} # times between nodes

    C = int(argv[1]) # vehicle capacity
    Q = {i: rnd.randint(1,C/2) for i in N} # amount that needs to be delivered to each customer i in N


    model = Model('CVRP')

    ########## Variables (mixed)
    x = model.addVars(A, vtype=GRB.BINARY) # binary variables (take or not take route to client)
    u = model.addVars(N, vtype=GRB.CONTINUOUS) # continuous variables for the mass of the product

    model.modelSense = GRB.MINIMIZE

    ###### Single objective (distance only)
    # model.setObjective(quicksum(x[i,j]*C[i,j] for i,j in A))

    ###### Multi objective (distance & time) - weighted
    model.setObjectiveN(quicksum(x[i,j]*D[i,j] for i,j in A), 0, weight=0.8)
    model.setObjectiveN(quicksum(x[i,j]*T[i,j] for i,j in A), 1, weight=0.2)

    ###### Multi objective (distance & time) - heirarchial approach
    # model.setObjectiveN(quicksum(x[i,j]*C[i,j] for i,j in A), 0, 1)
    # model.setObjectiveN(quicksum(x[i,j]*T[i,j] for i,j in A), 1, 0)


    ########## Constraints
    model.addConstrs(quicksum(x[i,j] for j in V if j!=i) == 1 for i in N) # entering is 1
    model.addConstrs(quicksum(x[i,j] for i in V if i!=j) == 1 for j in N) # exiting is 1

    # if a road to client j is taken from client i, then u[j] will be the
    # sum of the product mass until client i PLUS the mass that client j needs
    model.addConstrs((x[i,j] == 1) >> (u[i]+Q[j] == u[j]) 
                        for i,j in A if i!=0 and j!=0)

    # sum of product mass until node i is greater/equal to the product mass that i'th client needs
    model.addConstrs(u[i] >= Q[i] for i in N)
    # sum of product mass until node i is less/equal to the MAX capacity of the vehicle
    model.addConstrs(u[i] <= C for i in N)


    # stop if the solution is very close to optimal OR after 30 seconds
    model.Params.MIPGap = 0.1
    model.Params.TimeLimit = 30

    model.optimize()


    ########################## Plot optim results ##########################


    active_arcs = [a for a in A if x[a].x > 0.9] # by doing x[a].x we are accessing the solution (if the a'th bit was on or off)

    plt.plot(x_coord[0], y_coord[0], c='r', marker='s')
    plt.scatter(x_coord[1:], y_coord[1:], c='b')
    for i,j in active_arcs:
        plt.plot([x_coord[i], x_coord[j]], [y_coord[i], y_coord[j]], c='g', zorder=0)
    plt.show()


if __name__ == "__main__":
    main()