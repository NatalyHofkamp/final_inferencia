import numpy as np
from matplotlib import pyplot as plt

# --------------------- BUSCAR PARAMETRO DE DISTRIBUCION NORMAL ---------------------
N = 1000
mu_esp = 0
sigma_esp = 1
norms = np.random.normal(0, 1, size=N)

def find_best_param_norm(muestra, n):
    muestra = np.array(muestra)
    mu = sum(muestra)/n
    sigma = np.sqrt(sum((muestra - mu)**2)/n)
    
    return mu, sigma

mu_obs, sigma_obs = find_best_param_norm(norms, N)
print(f'La muestra debería tener mu = {mu_esp} y obtuvo: mu = {round(mu_obs, 2)}')
print(f'La muestra debería tener sigma = {sigma_esp} y obtuvo: sigma = {round(sigma_obs, 2)}')

# --------------------- EJERCICIO LIBRO (PAG 55) ---------------------
# EJERCICIO 1:
# (a) 1. plot p(x|lamb) versus x, for lamb = 1
lamb = 1
expos = np.arange(0, 10, 0.1)

def p_expo(x, lam):
    proba = 0
    if x >= 0:
        proba = lam*np.exp(-lam*x)
    return proba

plt.figure()
plt.scatter(expos, np.array([p_expo(x, lamb) for x in expos]))
plt.title('')
plt.xlabel('x')
plt.ylabel('p(x | lambda=1)')
plt.show()
plt.close()


# 2. plot p(x|lam) versus lamb (0 <= lamb <= 5, for x = 2
x = 2
lambs = np.arange(0, 5, 0.1)

def p_expo(x, lam):
    proba = 0
    if x >= 0:
        proba = lam*np.exp(-lam*x)
    return proba

plt.figure()
plt.scatter(lambs, np.array([p_expo(2, l) for l in lambs]))
plt.xlabel('lambda')
plt.ylabel('p(x=2 | lambda)')
plt.show()
plt.close()


# (b) Hacer un gráfico de calcular el lambda 

def plot_lamb_like(lam):
    size_m = np.arange(50, 100000, 50)
    means = []

    plt.figure()
    for s in size_m:
        means.append(1/np.mean(np.random.exponential(lam, s)))
    plt.plot(size_m, means)
    #plt.xticks(ticks=np.log10([1, 10, 100]), labels=[1, 10, 100])
    plt.xlabel('tamaño muestra')
    plt.ylabel('lambda likelihood')
    plt.show()
    plt.close()

plot_lamb_like(2)

lam_converge = 1 / np.mean(np.random.exponential(2, 100000))
print(f"El valor de lambda que converge es: {lam_converge}")


# EJERCICO 2:
# (b) 

# Este fragmento de código genera un gráfico de dispersión para visualizar 
# la función de densidad de probabilidad (PDF) condicional de una distribución uniforme en 
# el intervalo [0, 0.6]. Aquí está una explicación más detallada:
# unif_rand: Un array que va desde 0 hasta 1 con incrementos de 0.1. Este array 
# representa valores aleatorios generados de manera uniforme.
# unif_max: Un valor fijo de 0.6, que representa el valor máximo de la 
# distribución uniforme en este caso.
# p_unif(unif_max, x): Una función que calcula la probabilidad condicional p(x∣θ) para 
# la distribución uniforme en el intervalo [0, 0.6]. La probabilidad es constante en 
# este intervalo y se calcula como 1/θ para 
# x en el intervalo [0, 0.6], y 0 fuera de este intervalo.

unif_rand = np.arange(0, 1, 0.1)
unif_max = 0.6
def p_unif(unif_max, x):
    tita = 1/unif_max
    if x < 0 or x > 0.6:
        return 0
    return tita


plt.figure()
plt.scatter(unif_rand, np.array([p_unif(unif_max, u) for u in unif_rand]))
plt.title('')
plt.xlabel('x')
plt.ylabel('p(x | tita)')
plt.show()
plt.close()