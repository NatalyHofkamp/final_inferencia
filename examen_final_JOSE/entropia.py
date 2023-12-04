import random
from math import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# #---------------ENTROPIA -------------------
def H(p):
    return -p*np.log2(p)-(1-p)*log2(1-p)

def entropia_bernoulli():
    Hs = []
    ps = np.linspace(0, 1, 1000)
    for p in ps:
        if p != 0 and p != 1:
            Hs.append(H(p))
        else: 
            Hs.append(0)
    plt.plot(ps, Hs)
    plt.xlabel('Probability (p)')
    plt.ylabel('Entropy (H)')
    plt.title('Entropy of a Bernoulli Random Variable')
    plt.grid(True)
    plt.show()



#--------------CON LAS HOJAS / entropía conjunta ----------
def leer_csv(arch):
    """
    Lee un archivo CSV y organiza los datos en dos clases diferentes.

    Parameters:
        arch (str): Nombre del archivo CSV a leer.

    Returns:
        dict: Diccionario con los datos de la Clase 1 (largo y ancho).
        dict: Diccionario con los datos de la Clase 2 (largo y ancho).
    """
    clase1 = {'largo': [], 'ancho': []}
    clase2 = {'largo': [], 'ancho': []}

    with open(arch, 'r') as a:
        lines = a.readlines()
        for line in lines[1:]:
            line = line.rstrip()
            line = line.split(',')
            if line[0] == 'C1':
                clase1['largo'].append(float(line[1]))
                clase1['ancho'].append(float(line[2]))
            else:
                clase2['largo'].append(float(line[1]))
                clase2['ancho'].append(float(line[2]))

    return clase1, clase2

c1, c2 = leer_csv('dataset_hojas.csv')


def conj(l, a, bins):
    calc, l_b, a_b = np.histogram2d(l, a, bins)
    norm = calc/np.sum(calc)
    return norm, calc, l_b , a_b

norm, conjunta_dist, l_b ,a_b = conj(c1['largo'], c1['ancho'], 5)
#print(conjunta_dist)


#conjunta en relacion con la conj normalizada
def h_conj(prob):
    h = 0
    for row in prob:
        for j in row:
            if j != 0:  # log(0) es -inf
                h -= j * np.log2(j)
    return h



def probabilidad_conjunta(ancho, largo, bins):
    H, x_edges, y_edges = np.histogram2d(ancho, largo, bins)
    p = H / np.sum(H)  # Normaliza el histograma para obtener la distribución de probabilidad conjunta
    return H, p

prob_conjunta_c1 = probabilidad_conjunta(c1['largo'], c1['ancho'], 5)
#print("Probabilidad Conjunta (Clase 1):", prob_conjunta_c1)


#-----------------ENTROPIA CONDICIONAL----------------

def entropia_condicional(ancho,largo, bins):
    h = 0
    H, x_edges, y_edges = np.histogram2d(ancho,largo, bins, range=((min(ancho), max(ancho)),(min(largo), max(largo))))
    p = H / np.sum(H)  # Normaliza el histograma para obtener la distribución de probabilidad conjunta
    # Calcula la probabilidad de A(P(A)) → esto esta bueno para la demostracion
    p_A = np.sum(H, axis=0)
    # Calcula la probabilidad conjunta de L y A (ancho y largo)
    p_conju= H
    # Calcula la probabilidad condicional de L(largo) dado A(ancho)
    p_condi= p_conju/ p_A
    h= 0
    for x in range(len(p_condi)):
        for i in range(len(p_condi)):
                if p[x,i] != 0 and p_condi[x,i] !=0:
                    h-= p[x,i]*np.log2(p_condi[x,i])
    # print ("entropia condicional→",h)
    return h

entro_prob_condicional_c1 = entropia_condicional(c1['largo'], c1['ancho'], 5)
print("Entropia Probabilidad Condicional (Clase 1):", entro_prob_condicional_c1)


#teoría → demostracion de que la encontropía de la marginal + la condicional==  a la entropía conjunta
#parece importante→ condicional baja la entropía siempre y cuando las variables sean independientes
#esto demuestra que la conjunta > marginal 
def entropia (p):
     return -p*np.log2(p)

def entropia_marginal (p_m):
    entro_muestra= 0
    for x in p_m:
        if x != 0:
            entro_muestra += entropia(x)
    return entro_muestra


# Función para demostrar la propiedad de entropía condicional
def demostracion(ancho, largo, bins):
    entro_conjunta = probabilidad_conjunta(ancho, largo, bins)
    entro_ancho = probabilidad_conjunta(ancho, ancho, bins)  # Entropía marginal de ancho
    entro_condicional = entropia_condicional(ancho, largo, bins)

    if abs(entro_conjunta - (entro_ancho + entro_condicional)) <= 1e-4:
        print('Se cumple H(X, Y) = H(X) + H(Y|X)')
    else:
        print('No se cumple H(X, Y) = H(X) + H(Y|X)')

def demo_indep(ancho,largo, bins):
    H, x_edges, y_edges = np.histogram2d(ancho,largo, bins, range=((min(ancho), max(ancho)),(min(largo), max(largo))))
    e_conju = probabilidad_conjunta(ancho,largo, bins)
    margi_A= np.sum(H, axis=0)
    p_A = margi_A/np.sum(margi_A)
    entro_A= entropia_marginal(p_A)
    margi_L = np.sum(H,axis = 1)
    p_L = margi_L/np.sum(margi_L)
    entro_L = entropia_marginal(p_L)
    x = entro_A + entro_L
    
    if abs(entro_L - entro_A) < 1e-4:
        print('Son independientes')
    else:
        print('No son independientes')



def entropia_conjunta(prob_conjunta):
    p_nonzero = prob_conjunta[prob_conjunta != 0]  # Filtrar valores no nulos
    entropia = -np.sum(p_nonzero * np.log2(p_nonzero))  # Calcular la entropía
    return entropia

#-----------------ENTROPIA RELATIVA----------------
def calc_marginal(data, anch):
   p_x, _ = np.histogram(data, anch)
   p_x = p_x/len(data)
   return p_x

def mutual_info(title, conj, data1, data2, anch1, anch2):
    diver = 0
    p_x_y = conj
    p_x =  data1
    p_y = data2
    for i in range(anch1):
        for j in range(anch2):
            if p_x_y[i][j] !=0:
                diver += p_x_y[i][j]*(log2(p_x_y[i][j]/(p_x[i]*p_y[j])))

    print(title + str(diver))

norm, calc, l_b , a_b = conj(c1['largo'][:len(c2['largo'])], c2['largo'], 5)  
marg_1 = calc_marginal(c1['largo'][:len(c2['largo'])], 5)
marg_2 = calc_marginal(c2['largo'], 5)
mutual_info("Largos: " ,norm, marg_1, marg_2, 5, 5)


norm, calc, l_b , a_b = conj(c1['ancho'][:len(c2['ancho'])], c2['ancho'], 5)  
marg_1 = calc_marginal(c1['ancho'][:len(c2['ancho'])], 5)
marg_2 = calc_marginal(c2['ancho'], 5)
mutual_info("Anchos: " ,norm, marg_1, marg_2, 5, 5)



def f(x):
    return -np.log2(x)

def histo_densidad(muestra,bins):
    plt.xlabel(r'$Y$',fontsize = 15)
    plt.ylabel(r'Frecuencia relativa',fontsize = 15)
    plt.title("Histograma de densidad para ancho = "+str(bins)+" con "+str(len(muestra))+" elementos")
    plt.hist(muestra,density = True ,edgecolor = 'black')
    plt.show()

def main():
    muestra = np.random.uniform(low=0, high=1, size=1000)
    muestra2 = f(muestra)
    bins = 15
    histo_densidad(muestra,bins)
    histo_densidad(muestra2 ,bins)
    esperanza = np.mean(muestra)
    esperanza2 = np.mean(muestra2)
    print("E(x) →",esperanza)
    print("E[f(x)]→",esperanza2)
    print("f[E(x)]→",-np.log2(esperanza))

    print("Entropía:", entropia)


if __name__ == '__main__':
    main()


