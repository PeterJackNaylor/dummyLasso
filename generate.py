from simulations_X import options, produce_synthetic_data

from jax import random

key = random.PRNGKey(42)

print(key)
def formula(x, n=None):
    n = x.shape[0] if n is None else n

    x1 = x[:, 0]
    x2 = 2 * x[:, 1]
    x3 = 4 * x[:, 2]
    x4 = 8 * x[:, 3]
    key = random.PRNGKey(42)
    eps = random.normal(key, shape=(n,))

    y = x1 + x2 + x3 + x4 + eps
    return y


def main():
    opt = options()
    produce_synthetic_data(opt.n, opt.p, formula, correlated=False)


if __name__ == "__main__":
    main()
