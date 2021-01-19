import numpy as np
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.problem import Problem
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_problem
from pymoo.optimize import minimize

class travel_pro(Problem):

    def __init__(self):
        xl = 0* np.ones(10)
        xu = np.ones(10)
        super().__init__(n_var=10, n_obj=2, n_constr=1, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.sum((x - 0.5) ** 2, axis=1)
        out["G"] = 0.1 - out["F"]


problem =travel_pro()

# algorithm =GA(
#     pop_size=100,
#     eliminate_duplicates=True)

algorithm = NSGA2(pop_size=50)

res = minimize(problem,
               algorithm,
               ('n_gen', 100),
               seed=1,
               verbose=True)
#
# plot = Scatter()
# plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
# plot.add(res.F, color="red")
# plot.show()

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))


# class travel_pro(Problem):
#
#     def __init__(self):
#         super().__init__(n_var=10, n_obj=2, n_constr=1, xl=0, xu=1)

    # def _evaluate(self, x, out, *args, **kwargs):
    #     out["F"] = np.sum((x - 0.5) ** 2, axis=1)
    #     out["G"] = 0.1 - out["F"]


# problem = get_problem("g01")
# algorithm = GA(pop_size=5)
# res = minimize(problem,
#                algorithm,
#                ('n_gen', 5),
#                verbose=True,
#                seed=1)