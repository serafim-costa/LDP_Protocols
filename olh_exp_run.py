from olh import OLH
import numpy as np

olh = OLH()
n_user = 1000
domain = 100000
ep = [0.1, 0.5, 1, 2, 5]
X = np.zeros(n_user, dtype=np.int)
for i in range(n_user):
    X[i] = np.random.randint(domain)

for e in ep:
    mse_mean, mse_std = olh.run(X, e, domain, n_user)
    print("Epsilon = {}: mse_mean={} mse_std={}".format(e, mse_mean, mse_std))