import fo
import numpy as np
from data import Data

class LDPComp:

    def __init__(self, repeat=1):
        self.repeat = repeat
        self.real_dist = []

    def calculate_true_dist(self, X, domain):
        results = np.zeros(domain, dtype=np.int)
        for i in range(len(X)):
            if len(X[i]) == 0:
                continue
            rand_index = np.random.randint(len(X[i]))
            value = X[i][rand_index]
            results[value] += 1
        return results

    def build_topk_result(self, est_dist, key_list, top_k):
        sorted_indices = np.argsort(est_dist)
        key_result = []
        value_result = []
        for j in sorted_indices[-top_k:]:
            key_result.append(key_list[j])
            value_result.append(est_dist[j])
        return key_result, value_result

    def error_metric(self, domain, real_dist, estimate_dist):
        abs_error = 0.0
        for x in range(domain):
            abs_error += np.abs(real_dist[x] - estimate_dist[x]) ** 2
        return abs_error / domain

    def run(self, X, domain, epsilon):
        self.real_dist = self.calculate_true_dist(X, domain)
        olh_est_dist = fo.lh(self.real_dist, epsilon)
        grr_est_dist = fo.rr(self.real_dist, epsilon)

        olh_error = self.error_metric(domain, self.real_dist, olh_est_dist)
        grr_error = self.error_metric(domain, self.real_dist, grr_est_dist)
        return olh_error, grr_error

    def run(self, X, domain, epsilon, k):
        real_dist = self.calculate_true_dist(X, domain)
        kv_real_dist = self.build_topk_result(real_dist, range(domain), k)

        olh_est_dist = fo.lh(real_dist, epsilon)
        freq_olh_error = self.error_metric(domain, real_dist, olh_est_dist)
        kv_olh_est_dist = self.build_topk_result(olh_est_dist, range(domain), 2*k)

        grr_est_dist = fo.rr(real_dist, epsilon)
        freq_grr_error = self.error_metric(domain, real_dist, grr_est_dist)
        kv_grr_est_dist = self.build_topk_result(grr_est_dist, range(domain), 2*k)

        olh_error = len(np.intersect1d(kv_real_dist[0], kv_olh_est_dist[0]))
        grr_error = len(np.intersect1d(kv_real_dist[0], kv_grr_est_dist[0]))
        return olh_error, grr_error, freq_olh_error, freq_grr_error


def exp1():
    n_user = 5000
    domain_l = [50, 100, 200, 500, 1000, 2000]
    for domain in domain_l:
        X = [0] * n_user
        total_item = 0
        no_item = 0
        repeat = 20
        for i in range(n_user):
            item_count = np.random.randint(10)
            total_item += item_count
            if item_count == 0:
                no_item += 1
            items = np.random.randint(domain, size=item_count)
            X[i] = items

        ldp = LDPComp()
        # data = Data(dataname='kosarak', limit=1000)
        ep = 3
        k = 10
        olh_c = np.zeros(repeat)
        grr_c = np.zeros(repeat)
        olh_f = np.zeros(repeat)
        grr_f = np.zeros(repeat)
        for i in range(repeat):
            olh_c[i], grr_c[i], olh_f[i], grr_f[i] = ldp.run(X, domain, ep, k)
        print(
            "Domain = {:<6}: olh_cand_cout(%)= {:<7.2f} grr_cand_cout(%)= {:<7.2f} olh_mse= {:<8} grr_mse= {:<8}".format(
                domain, np.mean(olh_c) / float(k), np.mean(grr_c) / float(k),
                int(np.mean(olh_f)), int(np.mean(grr_f))))

def exp2(domain):
    n_user = 5000
    X = [0] * n_user
    total_item = 0
    no_item = 0
    repeat = 3
    for i in range(n_user):
        item_count = np.random.randint(10)
        total_item += item_count
        if item_count == 0:
            no_item += 1
        items = np.random.randint(domain, size=item_count)
        X[i] = items

    ldp = LDPComp()
    #data = Data(dataname='kosarak', limit=1000)
    ep = [0.1, 1, 2, 3, 4, 5, 6]
    k = int(domain / 10)
    print("domain={} k={}".format(domain, k))
    for e in ep:
        olh_c = np.zeros(repeat)
        grr_c = np.zeros(repeat)
        olh_f = np.zeros(repeat)
        grr_f = np.zeros(repeat)
        for i in range(repeat):
            olh_c[i], grr_c[i], olh_f[i], grr_f[i] = ldp.run(X, domain, e, k)
        print(
            "Epsilon = {:<6}: olh_cand_cout= {:<7.2f} grr_cand_cout= {:<7.2f} olh_mse= {:<8} grr_mse= {:<8}".format(
                e, np.mean(olh_c) , np.mean(grr_c),
                int(np.mean(olh_f)), int(np.mean(grr_f))))

def exp3():
    data = Data(dataname='kosarak', limit=1000)
    n_user = len(data.data)
    domain = 42178
    repeat = 10

    ldp = LDPComp()
    ep = [0.1, 1, 2, 3, 4, 5, 6]
    k = 4200
    for e in ep:
        olh_c = np.zeros(repeat)
        grr_c = np.zeros(repeat)
        olh_f = np.zeros(repeat)
        grr_f = np.zeros(repeat)
        for i in range(repeat):
            olh_c[i], grr_c[i], olh_f[i], grr_f[i] = ldp.run(data.data, domain, e, k)
        print(
            "Epsilon = {:<6}: olh_cand_cout(%)= {:<7.2f} grr_cand_cout(%)= {:<7.2f} olh_mse= {:<8} grr_mse= {:<8}".format(
                e, np.mean(olh_c) / float(k), np.mean(grr_c) / float(k),
                int(np.mean(olh_f)), int(np.mean(grr_f))))

if __name__ == "__main__":
    exp2(1000)
    exp2(5000)
    exp2(10000)
    exp2(20000)
