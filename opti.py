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

    def __init__(self, t0=5, t1=7, t2=3, d0=100, d1=50, d2=140):
        xl = 0* np.ones(3)
        xu = np.ones(3)
        super().__init__(n_var=3, n_obj=2, n_constr=6, xl=xl, xu=xu, type_var=np.int)

        self.t0=t0
        self.t1=t1
        self.t2=t2
        self.d0=d0
        self.d1=d1
        self.d2=d2

    def _evaluate(self, X, out, *args, **kwargs):
        # objectives
        f_time = X[:,0]*self.t0 + X[:,1]*self.t1 + X[:,2]*self.t2
        f_distance = X[:,0]*self.d0 + X[:,1]*self.d1 + X[:,2]*self.d2
        out["F"] = np.column_stack([f_time, f_distance])
        # out["F"] = f_distance

        # constraints
        g0 = X[:,0] + X[:,2] - 2 # x0 + x2 = 2
        g1 = X[:,1] + X[:,2] - 2 # x1 + x2 = 2
        g2 = X[:,1] + X[:,0] - 2 # xq + x0 = 2

        g3 = - X[:,0] - X[:,2] + 2 # x0 + x2 = 2
        g4 = - X[:,1] - X[:,2] + 2 # x1 + x2 = 2
        g5 = - X[:,1] - X[:,0] + 2 # xq + x0 = 2

        out["G"] = anp.column_stack([g0, g1, g2, g3, g4, g5])

problem = travel_pro()

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
               ('n_gen', 100),
               verbose=True)

################### Print solutions:

print("Best solution found: %s" % res.X)
print("Function value: %s" % res.F)
print("Constraint violation: %s" % res.CV)

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, color="red")
plot.show()
# Scatter().add(res.F).show()
