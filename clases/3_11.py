import random
import numpy as np
import matplotlib.pyplot as plt
import math


def calc_energia(syst):
    energia = 0
    for j in range(len(syst)):
        for i in range(len(syst)):
            value = 0
            actual = syst [j][i]
            for x in [-1,1]:
                if 0<= i+x <len(syst):
                    value+= actual * syst[j][i+x]
                if 0<= j+x <len(syst):
                    value+= actual * syst[j+x][i]
            energia += value
    return (-0.5)*energia

def generate_syst(size):
    return np.random.choice([-1, 1], size=(size, size))

def plot_syst(syst):
     print(np.array(syst))

def change_polo(syst,k,t):
    energia_actual = calc_energia(syst)
    pos = np.random.randint(0,10,2)
    syst[pos[0],pos[1]] = -syst[pos[0],pos[1]]
    new_energia = calc_energia(syst)
    if (new_energia < energia_actual):
        pass
    else : 
        prob =  np.exp(-np.abs(new_energia-energia_actual)/(k*t))
        if prob > np.random.random() :
            pass
        else:
            syst[pos[0],pos[1]] = -syst[pos[0],pos[1]]

def main():
    system = generate_syst(10)
    temp = 1
    conv_vals = []
    k_values = [0.5,1,4,5]
    for k in k_values:
        mag= []
        prom_acum = []
        for _ in range(10000):
            change_polo (system,k,temp)
            mag.append(np.sum(system))
            prom_acum.append(np.mean(mag))
        conv_vals.append(prom_acum)

    for i in range(len(k_values)) :
        plt.plot(range(len(conv_vals[i])),conv_vals[i],label = f'k = {k_values[i]}')
        plt.legend()
    plt.xlabel('Número de pasos de Monte Carlo')
    plt.ylabel('Magnetización promedio')
    plt.title('Evolución de la Magnetización Promedio en función de k')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()