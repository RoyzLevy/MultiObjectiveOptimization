import sys
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import Model, GRB, quicksum

def main():

    argv = sys.argv[1:] # number of clients, vehicle capacity, seed
    if len(argv) == 0:
        argv[0] = 10
        argv[1] = 10
        argv[2] = 1


    rnd = np.random
    rnd.seed(int(argv[2]))

    n = int(argv[0]) # number of clients

    x_coord = rnd.rand(n+1)*200
    y_coord = rnd.rand(n+1)*100


    plt.plot(x_coord[0], y_coord[0], c='r', marker='s')
    plt.scatter(x_coord[1:], y_coord[1:], c='b')
    plt.show()

    N = [i for i in range(1,n+1)] # clients
    V = [0] + N # factory + clients

    A = [(i,j) for i in V for j in V if i!=j] # arcs

    C = {(i,j): np.hypot(x_coord[i] - x_coord[j], y_coord[i] - y_coord[j]) for i,j in A} # distances between nodes

    Q = int(argv[1]) # vehicle capacity
    q = {i: rnd.randint(1,Q/2) for i in N} # amount that needs to be delivered to each customer i in N


    model = Model('CVRP')

    ###### Variables
    x = model.addVars(A, vtype=GRB.BINARY) # add variables to the model
    u = model.addVars(N, vtype=GRB.CONTINUOUS)

    ###### Objective
    model.modelSense = GRB.MINIMIZE
    model.setObjective(quicksum(x[i,j]*C[i,j] for i,j in A))

    ###### Constraints
    model.addConstrs(quicksum(x[i,j] for j in V if j!=i) == 1 for i in N) # entering is 1
    model.addConstrs(quicksum(x[i,j] for i in V if i!=j) == 1 for j in N) # exiting is 1


    model.addConstrs((x[i,j] == 1) >> (u[i]+q[j] == u[j]) 
                        for i,j in A if i!=0 and j!=0)

    model.addConstrs(u[i] >= q[i] for i in N)
    model.addConstrs(u[i] <= Q for i in N)


    # stop if the solution is very close to optimal OR after 30 seconds
    model.Params.MIPGap = 0.1
    model.Params.TimeLimit = 30

    model.optimize()

    active_arcs = [a for a in A if x[a].x > 0.9] # by doing x[a].x we are accessing the solution (if the a'th bit was on or off)

    plt.plot(x_coord[0], y_coord[0], c='r', marker='s')
    plt.scatter(x_coord[1:], y_coord[1:], c='b')
    for i,j in active_arcs:
        plt.plot([x_coord[i], x_coord[j]], [y_coord[i], y_coord[j]], c='g', zorder=0)
    plt.show()

main()