from __future__ import print_function

import torch


if __name__ == '__main__':
    x = torch.zeros(5, 3)
    y = torch.ones(5, 3)
    z = torch.empty(5, 3)
    torch.add(x, y, out=z)
    print(z)
