import cvxpy as cvx
import numpy as np

n_long=5
n_short=5
n_flat=5

n_all = n_long + n_short + n_flat

weights_current = [0.1] * n_long +[-0.1] * n_short + [0.0] * n_flat
betas = [0.8]*n_long + [1.2] * n_short + [0.8] * n_flat # changing code here

"""
two ways to define the weights variables
"""
#w_longs = cvx.Variable(n_long)
#w_shorts = cvx.Variable(n_short)
#w_flat = cvx.Variable(n_flat)  # add one more variable for new stocks
#w = cvx.vstack(w_longs, w_shorts , w_flat) # stack 3 variables together

w = cvx.Variable(n_all)
w_longs =w[0:n_long]
w_shorts =w[n_long:n_long+n_short]
w_flat = w[n_long+n_short:]

"""
objective is to minimize the variance of the weights variables
"""
objective = cvx.Minimize(cvx.sum_squares(w-np.mean(w))) # objective is to minimize the variance of the weights 

"""
I kept what the questions gave, add beta nutral & new stocks constrains
"""
con_bounds = [w_longs>=0.0,
              w_shorts<=-0.0,
              w_longs<=1.0,
              w_shorts >= -1.0,
              w_flat >= -1.0,  # additional constrain
              w_flat <= 1.0,   # additional constrain
              w.T*betas <=0.2]  # additional constrain

con_gmv = [cvx.sum_entries(cvx.abs(w_longs)) +
                    cvx.sum_entries(cvx.abs(w_shorts)) <= 3.0]
con_nmv = [cvx.abs(cvx.sum_entries(w_longs)+cvx.sum_entries(w_shorts)) <= 0.2]
constraints=con_gmv+con_nmv+con_bounds

# Form and solve problem.
cvxopt = cvx.Problem(objective, constraints)

result = cvxopt.solve()

weight = np.squeeze(np.asarray(w.value))
print(cvxopt.status, weight)