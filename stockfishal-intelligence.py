import matplotlib.pyplot as plt
from stockfish import Stockfish
import torch
import torch.nn as nn
import torch.optim as optim
import random

with open(r"stockfishdir.txt",encoding="utf-8") as file:
    dir = file.readline()

sf = Stockfish(dir)

start = [
    [0,0,0,1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0,1,0,0],
]

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
else:
    print("CPU")
    device = torch.device("cpu")

print(f"Device: {device}")


model = nn.Sequential(
    nn.Linear(768, 1536),
    nn.ReLU(),
    nn.Linear(1536, 1536),
    nn.ReLU(),
    nn.Linear(1536, 1536),
    nn.ReLU(),
    nn.Linear(1536, 1536),
    nn.ReLU(),
    nn.Linear(1536, 768),
    nn.ReLU(),
    nn.Linear(768, 768),
    nn.ReLU(),
    nn.Linear(768, 768),
    nn.ReLU(),
    nn.Linear(768, 384),
    nn.ReLU(),
    nn.Linear(384, 192),
    nn.ReLU(),
    nn.Linear(192, 96),
    nn.ReLU(),
    nn.Linear(96, 48),
    nn.ReLU(),
    nn.Linear(48, 32)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def getNumber(x: str, y: str) -> int: #turns ex: 'e2' into 11 so can be used like current[getNumber('e','2')]
    y1 = int(y)
    c = (y1 - 1) * 8
    match str(x):
        case "a":
            return c
        case "b":
            return c + 1
        case "c":
            return c + 2
        case "d":
            return c + 3
        case "e":
            return c + 4
        case "f":
            return c + 5
        case "g":
            return c + 6
        case "h":
            return c + 7


def makeMove(move: str, curr: list) -> list:
    x = list(move)
    try: #in case of promotion
        if curr[getNumber(x[0],x[1])] == [1,0,0,0,0,0,0,0,0,0,0,0]:
            match x[4]:
                case 'q':
                    curr[getNumber(x[2], x[3])] = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
                case 'r':
                    curr[getNumber(x[2], x[3])] = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
                case 'b':
                    curr[getNumber(x[2], x[3])] = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                case 'n':
                    curr[getNumber(x[2], x[3])] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            match x[4]:
                case 'q':
                    curr[getNumber(x[2], x[3])] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
                case 'r':
                    curr[getNumber(x[2], x[3])] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
                case 'b':
                    curr[getNumber(x[2], x[3])] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
                case 'n':
                    curr[getNumber(x[2], x[3])] = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    except:
        match move: #castling
            case 'e1g1':
                if curr[getNumber(x[0], x[1])] == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]:
                    curr[5] = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    curr[7] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            case 'e1c1':
                if curr[getNumber(x[0], x[1])] == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]:
                    curr[3] = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    curr[0] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            case 'e8g8':
                if curr[getNumber(x[0], x[1])] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]:
                    curr[61] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
                    curr[63] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            case 'e8c8':
                if curr[getNumber(x[0], x[1])] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]:
                    curr[58] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
                    curr[56] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        curr[getNumber(x[2], x[3])] = curr[getNumber(x[0], x[1])]
    curr[getNumber(x[0], x[1])] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]




loss = True
for epoch in range(100):
    if loss:
        loss = False
        turn = bool(random.getrandbits(1))
        current = start
        sf.make_moves_from_start()
        if turn:
            stockMove = sf.get_best_move(wtime=1000,btime=1000)
            current = makeMove(stockMove, current)

    inputs = "placeholder"
    targets = "placeholder"

    outputs = model(inputs)
    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



