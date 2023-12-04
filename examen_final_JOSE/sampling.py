import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import random

# ------------- SAMPPLING CON FUNC PROBA ------------- 
def f(x):
    """
    Calcula el cuadrado de un número.

    Parameters:
    - x (float): Número para el cual se calculará el cuadrado.

    Returns:
    float: Cuadrado de x.
    """
    return x**2

def muestra_norm(e, s):
    """
    Genera 10,000 muestras de una distribución normal con media e y desviación estándar s,
    recortando las muestras para que estén en el rango de 0 a 3. Devuelve las primeras 1000 muestras.

    Parameters:
    - e (float): Media de la distribución normal.
    - s (float): Desviación estándar de la distribución normal.

    Returns:
    numpy.ndarray: Array con las primeras 1000 muestras recortadas.
    """
        
    muestra = np.random.normal(e, s, 10000)
    rango_min = 0
    rango_max = 3
    muestra_recortada = np.clip(muestra, rango_min, rango_max)
    return np.array(muestra_recortada[:1000])

def graph_todo(f, p, q, fp, x1, x2):
    plt.figure()
    plt.scatter(x1, f, label='f(x)')
    plt.scatter(x2, p, label='p(x)')
    plt.scatter(x2, q, label='q(x)')
    plt.scatter(x2, fp, label='f(x)p(x)')
    plt.legend()
    plt.show()
    plt.close()


# 1. crear muestras de f(x) a partir de x que sigure p(x) y calc su estimación de E_p[f(x)]
esp_p = 0
std_p = 1
def p(x):
    return norm.pdf(x, loc=esp_p, scale=std_p)

norms_p = muestra_norm(esp_p, std_p)
sample_p = [f(x) for x in norms_p]

plt.figure()
plt.scatter(norms_p, sample_p, label='f(x) con x ~ p(x)')
plt.legend()
plt.show()
plt.close()

print(f"E_p[f(x)] = {round(np.mean(sample_p), 3)}")


# 2. graficar f(x), p(x) y f(x)p(x)
x1 = np.linspace(0, 3, 1000)
p_x = [p(x) for x in x1]
f_x = [f(x) for x in x1]
pf_x = np.array(p_x)*np.array(f_x)

plt.figure(figsize=(12, 7))
plt.plot(x1, f_x, label='f(x)')
plt.plot(x1, p_x, label='p(x)')
plt.plot(x1, pf_x, label='p(x)f(x)')
plt.legend()
plt.show()
plt.close()


# 3. crear dist q(x) y samplear x con esa distribucion (q(x) > p(x)f(x))
esp_q = 1.9
std_q = 0.8

def q(x):
    return norm.pdf(x, loc=esp_q, scale=std_q)
    
q_x = [q(x) for x in x1]

#grafico q(x) vs p(x)f(x)
plt.figure(figsize=(12, 7))
plt.plot(x1, q_x, label='q(x)')
plt.plot(x1, pf_x, label='p(x)f(x)')
plt.legend()
plt.show()
plt.close()

#estimación de Eq[p(x)f(x)/q(x)]
norms_q = muestra_norm(esp_q, std_q)
sample_q = [p(x)*f(x)/q(x) for x in norms_q]
print(f"Eq[p(x)f(x)/q(x)] = {round(np.mean(sample_q), 3)}")


# 4. comparar gráfico de f(x) con x ~ p(x) vs x ~ q(x)
plt.figure()
plt.subplot(1, 2, 1) 
plt.scatter(norms_p, sample_p, label='f(x) con x ~ p(x)')
plt.title('Gráfico sampleo con p(x)')
plt.legend()

plt.subplot(1, 2, 2)  
plt.scatter(norms_q, [f(x) for x in norms_q], label='f(x) con x ~ q(x)')
plt.title('Gráfico sampleo con q(x)')
plt.legend()

plt.tight_layout()
plt.show()

# hacer varias veces para ver difernecia ne varianza de promedio:
prom_p = []
prom_q = []
for _ in range(30):
    norms_p = muestra_norm(esp_p, std_p)
    sample_p = [f(x) for x in norms_p]
    prom_p.append(np.mean(sample_p))

    norms_q = muestra_norm(esp_q, std_q)
    sample_q = [p(x)*f(x)/q(x) for x in norms_q]
    prom_q.append(np.mean(sample_q))

print(f"Esperanza p: {np.mean(prom_p)} \nVarianza p: {np.var(prom_p)}")
print(f"\n \nEsperanza q: {np.mean(prom_q)} \nVarianza q: {np.var(prom_q)}")

