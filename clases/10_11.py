import numpy as np
from matplotlib import pyplot as plt
import random



def graph_tray(pos):
    plt.figure()
    t = np.linspace(0, 10, 1000)
    plt.plot(t, pos)
    plt.xlabel('t')
    plt.ylabel('posición')
    plt.show()

def graph_vel(vel):
    plt.figure()
    t = np.linspace(0, 10, 1000)
    plt.plot(t, vel)
    plt.xlabel('t')
    plt.ylabel('velocidad')
    plt.show()
    
def x(delta_t, x_k, v_k, sigma_a, error=True):
    pos = x_k + v_k*delta_t
    vel = v_k
    if error:
       a = np.random.normal(loc=0, scale=sigma_a, size=1)[0]
       pos += 0.5*a*delta_t**2
       vel += a*delta_t

    return pos, vel


def main():
    delta_t = 1
    v_k = 10
    sigma_a = 1
    pos = []
    vel = []
    for i in range(1000):
        p, v = x(delta_t, i, v_k, sigma_a)
        pos.append(p)
        vel.append(v)
    graph_tray(pos)
    graph_vel(vel)

if __name__ == '__main__':
    main()