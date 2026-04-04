import matplotlib.pyplot as plt
from stockfish import Stockfish
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

with open(r"stockfishdir.txt", "r",encoding="utf-8") as file:
    stockDir = file.readline()
sf = Stockfish(path=stockDir,depth=18)
#I'm not sure why but your code isn't working.

start = [
    [0,0,0,1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0,1,0,0]
]

class Piece:
    Empty: list = [0,0,0,0,0,0,0,0,0,0,0,0]
    WhitePawn: list = [1,0,0,0,0,0,0,0,0,0,0,0]
    WhiteKnight: list = [0,1,0,0,0,0,0,0,0,0,0,0]
    WhiteBishop: list = [0,0,1,0,0,0,0,0,0,0,0,0]
    WhiteRook: list = [0,0,0,1,0,0,0,0,0,0,0,0]
    WhiteQueen: list = [0,0,0,0,1,0,0,0,0,0,0,0]
    WhiteKing: list = [0,0,0,0,0,1,0,0,0,0,0,0]
    BlackPawn: list = [0,0,0,0,0,0,1,0,0,0,0]
    BlackKnight: list = [0,0,0,0,0,0,0,1,0,0,0,0]
    BlackBishop: list = [0,0,0,0,0,0,0,0,1,0,0,0]
    BlackRook: list = [0,0,0,0,0,0,0,0,0,1,0,0]
    BlackQueen: list = [0,0,0,0,0,0,0,0,0,0,1,0]
    BlackKing: list = [0,0,0,0,0,0,0,0,0,0,0,1]

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
    nn.Linear(48, 36)
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
        case _:
            return 64


def makeMove(move: str, curr: list) -> list:
    global enPass
    x = list(move)
    try: #in case of promotion
        if curr[getNumber(x[0],x[1])] == Piece.WhitePawn:
            match x[4]:
                case 'q':
                    curr[getNumber(x[2], x[3])] = Piece.WhiteQueen
                case 'r':
                    curr[getNumber(x[2], x[3])] = Piece.WhiteRook
                case 'b':
                    curr[getNumber(x[2], x[3])] = Piece.WhiteBishop
                case 'n':
                    curr[getNumber(x[2], x[3])] = Piece.WhiteKnight
        else:
            match x[4]:
                case 'q':
                    curr[getNumber(x[2], x[3])] = Piece.BlackQueen
                case 'r':
                    curr[getNumber(x[2], x[3])] = Piece.BlackRook
                case 'b':
                    curr[getNumber(x[2], x[3])] = Piece.BlackBishop
                case 'n':
                    curr[getNumber(x[2], x[3])] = Piece.BlackKnight
    except:
        match move: #castling
            case 'e1g1':
                if curr[getNumber(x[0], x[1])] == Piece.WhiteKing:
                    curr[5] = Piece.WhiteRook
                    curr[7] = Piece.Empty
            case 'e1c1':
                if curr[getNumber(x[0], x[1])] == Piece.WhiteKing:
                    curr[3] = Piece.WhiteRook
                    curr[0] = Piece.Empty
            case 'e8g8':
                if curr[getNumber(x[0], x[1])] == Piece.BlackKing:
                    curr[61] = Piece.BlackRook
                    curr[63] = Piece.Empty
            case 'e8c8':
                if curr[getNumber(x[0], x[1])] == Piece.BlackKing:
                    curr[58] = Piece.BlackRook
                    curr[56] = Piece.Empty
        if x[1] == '2' and x[3] == '4' and curr[getNumber(x[0], x[1])] == Piece.WhitePawn: #enpassant
            enPass = [x[0],'3']
        if x[1] == '7' and x[3] == '5' and curr[getNumber(x[0], x[1])] == Piece.BlackPawn:
            enPass = [x[0], '6']
        if enPass == [x[2],x[3]]:
            if curr[getNumber(x[0], x[1])] == Piece.WhitePawn and x[3] == '6':
                curr[getNumber(x[2],'6')] = Piece.Empty
            if curr[getNumber(x[0], x[1])] == Piece.BlackPawn and x[3] == '3':
                curr[getNumber(x[2],'3')] = Piece.Empty
        curr[getNumber(x[2], x[3])] = curr[getNumber(x[0], x[1])]
    curr[getNumber(x[0], x[1])] = Piece.Empty
    return curr

def makeLetter(x: int) -> str:
    match x:
        case 0:
            return 'a'
        case 1:
            return 'b'
        case 2:
            return 'c'
        case 3:
            return 'd'
        case 4:
            return 'e'
        case 5:
            return 'f'
        case 6:
            return 'g'
        case 7:
            return 'h'
        case _:
            return 'null'

def makeNumber(x: str) -> int:
    match x:
        case 'a':
            return 0
        case 'b':
            return 1
        case 'c':
            return 2
        case 'd':
            return 3
        case 'e':
            return 4
        case 'f':
            return 5
        case 'g':
            return 6
        case 'h':
            return 7
        case _:
            return 8


def aiOutNotation(x1: list, x2: list, y1: list, y2: list, prom: list, curr: list) -> str:
    cx1 = makeLetter(x1.index(1))
    cx2 = makeLetter(x2.index(1))
    cy1 = str(y1.index(1) + 1)
    cy2 = str(y2.index(1) + 1)
    out = cx1 + cy1 + cx2 + cy2
    if cy1 == '7' and cy2 == '8' or cy1 == '2' and cy2 == '1':
        if curr[getNumber(cx1,cy1)] == Piece.WhitePawn:
            match prom:
                case [1, 0, 0, 0]:
                    out = out + 'q'
                case [0, 1, 0, 0]:
                    out = out + 'r'
                case [0, 0, 1, 0]:
                    out = out + 'b'
                case [0, 0, 0, 1]:
                    out = out + 'n'
    return out

def getTargetList(x: str) -> list:
    x = list(x)


lossdat = []
steps = []
step = 0
loss = 5
cepoch = 0

try:
    checkpoint = torch.load(f"checkpoint(epoch: 0).pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    model.train()
except:
    print("Fail to load model, starting from start")

for epoch in range(100):
    current = start
    while True:
        step += 1
        try:
            targets = sf.get_best_move(wtime=1000,btime=1000)
        except:
            break
        if targets is None:
            break
        inputs = []
        for i in current:
            inputs.extend(i)
        inputs = torch.Tensor(inputs)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lossdat.append(loss.item())
        steps.append(step)
    if epoch % 20 == 0:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss
        }
        torch.save(checkpoint, f'checkpoint(epoch: {epoch}).pth')
    cepoch = epoch

checkpoint = {
    "epoch": cepoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss
}
torch.save(checkpoint, f'checkpoint(epoch: {cepoch}).pth')

plt.plot(step, lossdat)
plt.title("Neural Net Loss")
plt.ylabel("Loss")
plt.xlabel("Step")
plt.show()