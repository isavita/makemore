import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

words = [w.lower() for w in open("assets/names.txt", "r").read().splitlines()]
N = torch.zeros((27,27), dtype=torch.int32)
chars = sorted(list(set(".abcdefghijklmnopqrstuvwxyz")))
s2i = {s:i for i,s in enumerate(chars)}
i2s = {i:s for s,i in s2i.items()}
block_size = 4

def build_dataset(words):
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
    return X, Y
    
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])
# Network setup
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 10), generator=g)
W1 = torch.randn((40, 200), generator=g)
b1 = torch.randn(200, generator=g)
W2 = torch.randn((200, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True

# Debug the number of parameters
print(sum(p.nelement() for p in parameters))

# lre = torch.linspace(-3, 0, 1000)
# lrs = 10**lre
for i in range(100000):
    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (32,))
    # forward pass
    emb = C[Xtr[ix]]
    h = torch.tanh(emb.view(-1, 40) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])

    # backward pass 
    for p in parameters:
        p.grad = None
    loss.backward()
    # update
    lr = 0.01 if i > 50000 else 0.1
    for p in parameters:
        p.data += -lr * p.grad

def validate():
    emb = C[Xdev]
    h = torch.tanh(emb.view(-1, 40) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ydev)
    print(loss)

def sample():
    for _ in range(20):
        out = []
        context = [0] * block_size
        while True:
            emb = C[torch.tensor([context])]
            h = torch.tanh(emb.view(1, -1) @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break
        print("".join(i2s[i] for i in out[:-1]))
sample()
validate()
# training, validation, test
# 80%, 10%, 10%
