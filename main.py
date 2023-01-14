import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = [w.lower() for w in open("assets/names.txt", "r").read().splitlines()]
N = torch.zeros((27,27), dtype=torch.int32)
chars = sorted(list(set(".abcdefghijklmnopqrstuvwxyz")))
s2i = {s:i for i,s in enumerate(chars)}
i2s = {i:s for s,i in s2i.items()}

block_size = 3
X, Y = [], []
for w in words:
    #print(w)
    context = [0] * block_size
    for ch in w + ".":
        ix = s2i[ch]
        X.append(context)
        Y.append(ix)
        #print(''.join(i2s[i] for i in context), "--->", i2s[ix])
        context = context[1:] + [ix]

X = torch.tensor(X)
Y = torch.tensor(Y)
# Network setup
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 2), generator=g)
W1 = torch.randn((6, 100), generator=g)
b1 = torch.randn(100, generator=g)
W2 = torch.randn((100, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True

# Debug the number of parameters
print(sum(p.nelement() for p in parameters))

# lre = torch.linspace(-3, 0, 1000)
# lrs = 10**lre
for i in range(10000):
    # minibatch construct
    ix = torch.randint(0, X.shape[0], (32,))
    # forward pass
    emb = C[X[ix]]
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y[ix])

    # backward pass 
    for p in parameters:
        p.grad = None
    loss.backward()
    # update
    lr = 0.1
    for p in parameters:
        p.data += -lr * p.grad


print(loss)
# training, validation, test
# 80%, 10%, 10%
