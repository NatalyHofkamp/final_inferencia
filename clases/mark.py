import numpy as np
def energia(matriz: np.array):
    # recorre las filas
    sumatoria = 0
    for row in range(matriz.shape[0]):
        for col in range(matriz.shape[1]):
            
            for x,y in [(-1,0),(1,0),(0,-1),(0,1)]:
                    if 0 <= row+x < matriz.shape[0] and 0 <= col+y < matriz.shape[1]:
                        # print("sumando ", matriz[row,col], matriz[row+x,col+y]) 
                        sumatoria += matriz[row,col] * matriz[row+x,col+y]
    
    return -0.5 * sumatoria


def magnetizacion(matriz: np.array):
    return np.sum(matriz)



import matplotlib.pyplot as plt


k=1 
Temp = 1.5

size = 10  # Number of rows
def tets(k,t,size,iters):

    # Generate a random matrix of ones and minus ones
    matrix = np.random.choice([-1, 1], size=(size, size))

    magnetismo = []
    for M in range(iters):
        E1 = energia(matrix)

        #cambio un dipolo random
        pos = np.random.randint(0,size,2)
        matrix[pos[0],pos[1]] = -matrix[pos[0],pos[1]]
        E2 = energia(matrix)
        
        if E2 < E1:
            pass
        else:
            p = np.exp(-np.abs(E2-E1)/(k*Temp))

            if np.random.random() < p:
                pass
            else:
                matrix[pos[0],pos[1]] = -matrix[pos[0],pos[1]]

        magnetismo.append(magnetizacion(matrix))

    t = []
    for M in range(1,len(magnetismo)):
        t.append(np.sum(magnetismo[:M])/M)

    plt.plot(range(len(t)),t)
    plt.show()


# tets(k=1,t=2,size=10,iters=10000)
# tets(k=1,t=0.5,size=10,iters=10000)
matrix = np.random.choice([1, 1], size=(size, size))
print(matrix)

print(energia(matrix))