import numpy as np
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.problem import Problem
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_problem, get_algorithm, get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
import autograd.numpy as anp
from pymoo.visualization.scatter import Scatter

################### Problem:

class travel_pro(Problem):

    def __init__(self, times, distances):
        xl = 0* np.ones(len(times))
        xu = np.ones(len(times))
        super().__init__(n_var=len(times), n_obj=2, n_constr=8, xl=xl, xu=xu, type_var=np.int)
        self.times=times
        self.distances=distances

    def _evaluate(self, X, out, *args, **kwargs):
        # objectives
        f_time = X[:,0]*self.times[0] + X[:,1]*self.times[1] + X[:,2]*self.times[2] + X[:,3]*self.times[3] + X[:,4]*self.times[4] + X[:,5]*self.times[5]
        f_distance = X[:,0]*self.distances[0] + X[:,1]*self.distances[1] + X[:,2]*self.distances[2] + X[:,3]*self.distances[3] + X[:,4]*self.distances[4] + X[:,5]*self.distances[5]
        out["F"] = np.column_stack([f_time, f_distance])
        # out["F"] = f_time

        # constraints
        g0 = X[:,0] + X[:,1] + X[:,4] - 2
        g1 = X[:,0] + X[:,3] + X[:,5] - 2
        g2 = X[:,5] + X[:,4] + X[:,2] - 2
        g3 = X[:,1] + X[:,3] + X[:,2] - 2

        g4 = -X[:,0] - X[:,1] - X[:,4] + 2
        g5 = -X[:,0] - X[:,3] - X[:,5] + 2 
        g6 = -X[:,5] - X[:,4] - X[:,2] + 2
        g7 = -X[:,1] - X[:,3] - X[:,2] + 2

        out["G"] = anp.column_stack([g0, g1, g2, g3, g4, g5, g6, g7])

times       = [5,7,3,10,2,1]
distances   = [100,50,140,120,80,90]

problem = travel_pro(times, distances)

################### Algorithm chosen:

algorithm = NSGA2(pop_size=10)
# algorithm = NSGA2(pop_size=100,
#                   sampling=get_sampling("bin_random"),
#                   crossover=get_crossover("bin_two_point"),
#                   mutation=get_mutation("bin_bitflip"),
#                   eliminate_duplicates=True)

# algorithm = GA(
#     pop_size=200,
#     sampling=get_sampling("bin_random"),
#     crossover=get_crossover("bin_hux"),
#     mutation=get_mutation("bin_bitflip"),
#     eliminate_duplicates=True)

################### Solve problem:

res = minimize(problem,
               algorithm,
               ('n_gen', 50),
               verbose=True)

################### Print solutions:

print("Best solution found: %s" % res.X)
print("Function value: %s" % res.F)
print("Constraint violation: %s" % res.CV)

# plot = Scatter()
# plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
# plot.add(res.F, color="red")
# plot.show()
# Scatter().add(res.F).show()
