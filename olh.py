import argparse
import math
import numpy as np
import xxhash


class OLH():
    def __init__(self, repeat=1):
        self.repeat = repeat
        self.real_dist = []
        self.estimate_dist = []

    def calculate_real_dist(self, X, n_user):
        for i in range(n_user):
            self.real_dist[X[i]] += 1

    def perturb(self, X, n_user, p, q, g):
        Y = np.zeros(n_user)
        for i in range(n_user):
            v = X[i]
            x = (xxhash.xxh32(str(v), seed=i).intdigest() % g)
            y = x

            p_sample = np.random.random_sample()
            # the following two are equivalent
            # if p_sample > p:
            #     while not y == x:
            #         y = np.random.randint(0, g)
            if p_sample > p - q:
                # perturb
                y = np.random.randint(0, g)
            Y[i] = y
        return Y

    def aggregate(self, Y, p, g, n_user, domain):
        self.estimate_dist = np.zeros(domain)
        for i in range(n_user):
            for v in range(domain):
                if Y[i] == (xxhash.xxh32(str(v), seed=i).intdigest() % g):
                    self.estimate_dist[v] += 1
        a = 1.0 * g / (p * g - 1)
        b = 1.0 * n_user / (p * g - 1)
        self.estimate_dist = a * self.estimate_dist - b


    def error_metric(self, domain):
        abs_error = 0.0
        for x in range(domain):
            abs_error += np.abs(self.real_dist[x] - self.estimate_dist[x]) ** 2
        return abs_error / domain


    def run(self, X, epsilon, domain, n_user):
        self.real_dist = np.zeros(domain)
        self.estimate_dist = np.zeros(domain)

        self.calculate_real_dist(X, n_user)
        g = int(round(math.exp(epsilon))) + 1
        p = math.exp(epsilon) / (math.exp(epsilon) + g - 1)
        q = 1.0 / (math.exp(epsilon) + g - 1)

        results = np.zeros(self.repeat)
        for i in range(self.repeat):
            Y = self.perturb(X, n_user, p, q, g)
            self.aggregate(Y, p, g, n_user, domain)
            results[i] = self.error_metric(domain)
        return np.mean(results), np.std(results)




