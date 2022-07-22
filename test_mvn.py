import torch

from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
from tqdm.auto import trange


D = 32
mu = torch.zeros(D)
# torch.rand: Generates a DxD tensor of uniform random numbers [0, 1)
# .tril() sets all the upper elements to 0
L = torch.rand(D, D).tril()

dist = MultivariateNormal(mu, scale_tril=L, validate_args=False)

n = 10000
# x_sample = torch.zeros((n, D))
# for i in trange(n-1):
#     x_sample[i] = dist.sample()
x_sample = dist.sample(sample_shape=torch.Size([n]))

print(f'Var of entire tensor: {x_sample.var():.3f}')

x_sample_t = torch.transpose(x_sample, 0, 1).cpu().detach().numpy()

for i in range(D):
    print(f'{x_sample_t[i].var():.3f}, ', end='')
print('\n')

# fig, ax = plt.subplots(2, 4)
#
# for i in range(D):
#     ax_i = ax[i % 2][i // 2]
#     ax_i.hist(x_sample_t[i], bins=100)
#     ax_i.set_title(f'Dim:{i}, '
#                    f'mean = {x_sample_t[i].mean():.3f}, '
#                    f'var = {x_sample_t[i].var():.3f}')
#
# plt.show()
