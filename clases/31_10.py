from pylab import *
import scipy.stats as stats
# El código del trabajo práctico 1 fue realizado junto con Lucía Fernández Vincent.
def dist_uni(min, max, size):
    return np.random.uniform(low=min, high=max, size=size)

def varianza(data):
    med = esperanza(data)
    values = [(v - med) ** 2 for v in data]
    return esperanza(values)

def esperanza(data):
    return sum(data) / len(data)


def N(t,muestras):
    #calcula la acumulada hasta un tiempo dado
    return sum(muestras[:t])

def x(N,t,h,muestras):
    #calcula el incremento entre dos tiempos
    return N(t+h,muestras)-N(t,muestras)

def get_reali(muestras,h):
    reali = [0]
    for t in range(len(muestras)):
        incremento = x(N, t, h, muestras) 
        reali.append(incremento)  # Acumula los incrementos
    return reali

def m(wiener):
    esperanzas = []
    for t in range (1,len(wiener[0])):
        tiempo = []
        for realizacion in wiener:
            tiempo.append(realizacion[t])
        #esperanza del valor de las trayectorias en un tiempo dado
        esperanzas.append(esperanza(tiempo))
    return esperanzas


def get_z1z2():
    U1 = dist_uni(0,1,10000)
    U2 = dist_uni(0,1,10000)
    Z1 = np.sqrt(-2*np.log(U1))*np.cos(2*np.pi*U2)
    Z2 = np.sqrt(-2*np.log(U1))*np.sin(2*np.pi*U2)
    return Z1,Z2

def histo_densidad (muestra,bins):
    plt.xlabel(r'$Y$',fontsize = 15)
    plt.ylabel(r'Frecuencia relativa',fontsize = 15)
    plt.title("Histograma de densidad para ancho = "+str(bins)+" con "+str(len(muestra))+" elementos")
    plt.hist(muestra, bins = np.arange(min(muestra),max(muestra)+bins, bins), weights = np.zeros(len(muestra))+1/len(muestra),density = True,edgecolor = 'black')
    

def transfo(Z1, Z2, matrix):
    # Aplica la transformación lineal a Z1 y Z2
    z1_new, z2_new = np.dot(matrix, np.vstack((Z1, Z2)))

    plt.scatter(Z1, Z2,label = 'original')
    plt.scatter(z1_new, z2_new,label = 'blanqueada')
    plt.legend()
    plt.figure()
    
    return z1_new, z2_new


def ejercicio2():
    print("--------EJERCICIO 2 (A)----------")
    Z1, Z2 = get_z1z2()
    matricita = np.array([[1, 1], [1, 2]])
    # Calcula los autovalores
    autovalores = np.linalg.eigvals(matricita)

    print("Autovalores de matricita:")
    print(autovalores)
    #matriz de blanqueamientol
    mat_blanq = np.linalg.inv(np.sqrt(np.diag(autovalores)))
    z1_new, z2_new = transfo(Z1, Z2, mat_blanq)

    # print("Matriz de correlación original:")
    # print(correlation_matrix)

    # print("Matriz de correlación después de la transformación:")
    # print(correlation_matrix_new)


ejercicio2()