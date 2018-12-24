# coding: utf-8

import copy
from matplotlib import pyplot as pl
from matplotlib import animation as ani

w = [0,0]
b = 0
eta = 0.5
data=[[(1,4),1],[(0.5,2),1],[(2,2.3), 1], [(1, 0.5), -1], [(2, 1), -1],[(4,1),-1],[(3.5,4),1],[(3,2.2),-1]]
record = []

def sign(vec):
    global w, b
    res = 0
    res = vec[1]*(w[0]*vec[0][0]+w[1]*vec[0][1]+b)
    if res > 0: return 1
    else: return -1

def update(vec):
    global w, b, record
    w[0] += eta*vec[1]*vec[0][0]
    w[1] += eta*vec[1]*vec[0][1]
    b += eta*vec[1]
    record.append([copy.copy(w), b])

def perceptron():
    count=1
    for ele in data:
        flag = sign(ele)
        if not flag > 0:
            count = 1
            update(ele)
        else:
            count += 1
    if count >= len(data):
        return 1
    return 0


if __name__ == "__main__":
    while 1:
        if perceptron() > 0:
            break
    print(record)

    x1 = []
    y1 = []
    x2 = []
    y2 = []

    fig = pl.figure()
    ax = pl.axes(xlim=(-1, 5), ylim=(-1, 5))
    line, = ax.plot([], [], 'g', lw=2)

    def init():
        line.set_data([], [])
        for p in data:
            if p[1] > 0:
                x1.append(p[0][0])
                y1.append(p[0][1])
            else:
                x2.append(p[0][0])
                y2.append(p[0][1])
        pl.plot(x1,y1,'or')
        pl.plot(x2,y2,'ob')
        return line,

    def animate(i):
        global record, ax, line
        w = record[i][0]
        b = record[i][1]
        x1 = -5
        '''y 的计算方式不理解'''
        y1 = -(b + w[0]*x1) / w[1]
        x2 = 6
        y2 = -(b + w[0]*x2)/w[1]
        line.set_data([x1, x2], [y1, y2])
        print(x1, y1, x2, y2)
        return line,

    animat = ani.FuncAnimation(fig, animate, init_func=init, frames=len(record), interval=1000, repeat=True, blit=True)
    pl.show()
    # animat.save('')

