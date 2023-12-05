import numpy as np
from matplotlib import pyplot as plt
import random
from scipy.stats import pearsonr

#CLASE 1:
def leer_csv(arch):
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

def calc_esp(data):
    return np.mean(data)

def calc_var(data):
    return np.var(data) #insesgado


#leer archivo: 
c1, c2 = leer_csv('dataset_hojas.csv')

# largo:
esp_largo_c1 = calc_esp(c1['largo'])
esp_largo_c2 = calc_esp(c2['largo'])

var_largo_c1 = calc_var(c1['largo'])
var_largo_c2 = calc_var(c2['largo'])
# print('Largo  Esperanza             Varianza')
# print('C1    ',esp_largo_c1, '  ', var_largo_c1 )
# print('C2    ',esp_largo_c2, '  ', var_largo_c2 )

#ancho:
esp_ancho_c1 = calc_esp(c1['ancho'])
esp_ancho_c2 = calc_esp(c2['ancho'])

var_ancho_c1 = calc_var(c1['ancho'])
var_ancho_c2 = calc_var(c2['ancho'])
# print('Ancho  Esperanza             Varianza')
# print('C1    ',esp_ancho_c1, '  ', var_ancho_c1 )
# print('C2    ',esp_ancho_c2, '  ', var_ancho_c2 )


# histogramas:
def make_hist(muestra1, ancho, title, x_label, type='frec'):
    if type == 'proba':
        plt.hist(muestra1, bins=np.arange(min(muestra1), max(muestra1) + ancho, ancho), weights=np.zeros(len(muestra1)) + 1. / len(muestra1), edgecolor='black')

    else:
        plt.hist(muestra1, bins=np.arange(min(muestra1), max(muestra1) + ancho, ancho), edgecolor='black')

    plt.xlabel(x_label)
    plt.ylabel('Frecuencia')
    plt.title(title)
    #plt.show()

def make_hist_2data(muestra1, muestra2, ancho, title, x_label, type='frec'):
    if type == 'proba':
        plt.hist(muestra1, bins=np.arange(min(muestra1), max(muestra1) + ancho, ancho), weights=np.zeros(len(muestra1)) + 1. / len(muestra1), edgecolor='black', alpha=0.75, label='clase1')
        plt.hist(muestra2, bins=np.arange(min(muestra2), max(muestra2) + ancho, ancho), weights=np.zeros(len(muestra2)) + 1. / len(muestra2), edgecolor='black', alpha=0.75,label='clase2')
    else:
        plt.hist(muestra1, bins=np.arange(min(muestra1), max(muestra1) + ancho, ancho), edgecolor='black', alpha=0.75,label='clase1')
        plt.hist(muestra2, bins=np.arange(min(muestra2), max(muestra2) + ancho, ancho), edgecolor='black',alpha=0.75, label='clase2')
    plt.xlabel(x_label)
    plt.ylabel('Frecuencia')
    plt.title(title)
    plt.legend()
    plt.show()

# make_hist(c1['largo'], 5, "Histograma proba de largo clase 1", 'Largo', type='proba')
# make_hist(c2['largo'], 5, "Histograma proba de largo clase 1", 'Largo',  type='proba')


# make_hist_2data(c1['largo'], c2['largo'], 5, "Histograma proba de largo ambas clases", 'Largo', type='proba')
# make_hist_2data(c1['ancho'], c2['ancho'], 3, "Histograma proba de ancho ambas clases", 'ancho', type='proba')

# make_hist_2data(c1['largo'], c2['largo'], 5, "Histograma frec de largo ambas clases", 'Largo')
# make_hist_2data(c1['ancho'], c2['ancho'], 3, "Histograma frec de ancho ambas clases", 'ancho')


#CLASE 2:

#covarianza insesgada
def calc_cov(data1, data2):
    l_prom = np.mean(data1)
    a_prom = np.mean(data2)

    n = len(data1)
    cov = 0
    for i in range(n):
        cov += (data1[i] - l_prom)*(data2[i] - a_prom)
    return cov/(n-1)


cov_clase1 = calc_cov(c1['largo'], c1['ancho'])
# print("Covarianza clase 1: ", cov_clase1)
# print("con numpy: ", np.cov(c1['largo'], c1['ancho'])) # insesgado

cov_clase2 = calc_cov(c2['largo'], c2['ancho'])
# print("Covarianza clase 2: ", cov_clase2)
# print("con numpy: ", np.cov(c2['largo'], c2['ancho'])) # insesgado


#coeficiente de correlacion insesgada
def calc_coef(data1, data2, cov):
    l_std= np.std(data1)
    a_std = np.std(data2)

    return cov/(l_std*a_std)


coef_clase1 = calc_coef(c1['largo'], c1['ancho'], cov_clase1)
# print("coeficiente de correlacion clase 1: ", coef_clase1)

coef_clase2 = calc_coef(c2['largo'], c2['ancho'], cov_clase2)
# print("coeficiente de correlacion clase 2: ", coef_clase2)

#con pearsonr:
# print('calc con numpy clase 1:', pearsonr(c1['largo'], c1['ancho'])[0], ', con p =', pearsonr(c1['largo'], c1['ancho'])[1])
# print('calc con numpy clase 2:', pearsonr(c2['largo'], c2['ancho'])[0],  'con p =', pearsonr(c2['largo'], c2['ancho'])[1])

#es alto este coeficiente? Para verificarlo se hacen permutaciones del dataset, 
# si se mantiene el coeficiente significa que no es alto...

def perms(data_l, data_a):
    coefs = []
    for _ in range(1000):
        new_a = random.sample(data_a, len(data_a))
        coefs.append(calc_coef(data_l, new_a, calc_cov(data_l, new_a)))

    return coefs

#perms clase 1
coefs1 = perms(c1['largo'], c1['ancho'])
coefs1.append(coef_clase1)
# make_hist(coefs1, 0.01, 'Histograma de coeficientes de correlacion al permutar clase 1', 
#           'coeficiente de correlacion', type='proba')

#perms clase 2
coefs2 = perms(c2['largo'], c2['ancho'])
coefs2.append(coef_clase2)
# make_hist(coefs2, 0.01, 'Histograma de coeficientes de correlacion al permutar clase 2', 
#           'coeficiente de correlacion', type='proba')