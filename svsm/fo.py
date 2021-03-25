import numpy as np
import xxhash


def oue(real_dist, eps):
    ee = np.exp(eps)
    p = 0.5
    q = 1 / ee + 1
    domain = len(real_dist)
    noisy_samples = oue_perturb(real_dist, p, q)
    est_dist = oue_aggregate(noisy_samples, domain, p, q, sum(real_dist))
    return est_dist


def oue_perturb(real_dist, p, q):

    domain = len(real_dist)
    n = sum(real_dist)
    perturbed_data = np.zeros(n, dtype=object)

    for k, v in enumerate(real_dist):
        k_binary = np.zeros(domain)
        k_binary[k] = 1
        for _ in range(v):
            p_sample = np.random.random_sample()
            q_sample = np.random.random_sample()
            for j in range(domain):
                if k_binary[j] == 0 and q_sample <= q:
                    k_binary[j] = 1
                elif k_binary[j] == 1 and q_sample <= q:
                    k_binary[j] = 0
            perturbed_data[j] += k_binary[j]
    return perturbed_data


def oue_aggregate(perturbed_data, domain, p, q, n):
    est_dist = np.zeros(domain, dtype=np.int32)
    for k, v in enumerate(perturbed_data):
        k_freq = (v / n) - q / (p - q)
        est_dist[k] = k_freq
    return est_dist


def lh(real_dist, eps):
    p = 0.5
    g = np.ceil(np.exp(eps) + 1)
    q = 1 / g
    domain = len(real_dist)

    noisy_samples = lh_perturb(real_dist, g, p)
    est_dist = lh_aggregate(noisy_samples, domain, g, p, q)

    return est_dist


def lh_perturb(real_dist, g, p):
    n = sum(real_dist)
    noisy_samples = np.zeros(n, dtype=object)
    samples_one = np.random.random_sample(n)
    seeds = np.random.randint(0, n, n)

    counter = 0
    for k, v in enumerate(real_dist):
        for _ in range(v):
            y = x = xxhash.xxh32(str(int(k)), seed=seeds[counter]).intdigest() % g

            if samples_one[counter] > p:
                y = np.random.randint(0, g - 1)
                if y >= x:
                    y += 1
            noisy_samples[counter] = tuple([y, seeds[counter]])
            counter += 1
    return noisy_samples


def lh_aggregate(noisy_samples, domain, g, p, q):
    n = len(noisy_samples)

    est = np.zeros(domain, dtype=np.int32)
    for i in range(n):
        for v in range(domain):
            x = xxhash.xxh32(str(v), seed=noisy_samples[i][1]).intdigest() % g
            if noisy_samples[i][0] == x:
                est[v] += 1

    # a = 1.0 / (p - q)
    # b = n * q / (p - q)

    a = 1.0 / (p - 1/g)
    b = n/g / (p - 1/g)
    est = a * est - b

    return est


def rr(real_dist, eps):
    domain = len(real_dist)
    ee = np.exp(eps)

    p = ee / (ee + domain - 1)
    q = 1 / (ee + domain - 1)

    noisy_samples = rr_perturb(real_dist, domain, p)
    est_dist = rr_aggregate(noisy_samples, domain, p, q)

    return est_dist


def rr_perturb(real_dist, domain, p):
    n = sum(real_dist)  # number of users
    perturbed_datas = np.zeros(n, dtype=np.int)
    counter = 0
    for k, v in enumerate(real_dist):
        for _ in range(v):
            y = x = k
            p_sample = np.random.random_sample()

            if p_sample > p:
                y = np.random.randint(0, domain - 1)
                if y >= x:
                    y += 1
            perturbed_datas[counter] = y
            counter += 1
    return perturbed_datas


def rr_aggregate(noisy_samples, domain, p, q):
    n = len(noisy_samples)

    est = np.zeros(domain)
    unique, counts = np.unique(noisy_samples, return_counts=True)
    for i in range(len(unique)):
        est[unique[i]] = counts[i]

    a = 1.0 / (p - q)
    b = n * q / (p - q)
    est = a * est - b

    return est
