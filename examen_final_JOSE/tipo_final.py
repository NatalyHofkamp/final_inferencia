import numpy as np
import matplotlib.pyplot as plt




# Estimar la entropia de una distribución uniforme U(-3,3) y 
# de una distribución gaussiana N(0,1).
N = 2000
bins = 12  # Número de bins para el histograma

def H(X, n_bins):
    hist, bins_edge = np.histogram(X, bins=n_bins, density=True)
    bin_width = bins_edge[1] - bins_edge[0]
    px = hist * bin_width
    entropy = -np.sum(px * np.log2(px[px != 0]))
    return entropy

# Estimar la entropía de la distribución uniforme U(-3,3)
muestras_unif = np.random.uniform(-3, 3, N)
H_unif = H(muestras_unif, bins)
print("Entropía de la distribución uniforme U(-3,3): ", H_unif)

# Estimar la entropía de la distribución gaussiana N(0,1)
muestras_norm = np.random.normal(0, 1, N)
H_norm = H(muestras_norm, bins)
print("Entropía de la distribución gaussiana N(0,1): ", H_norm)

# Visualizar los histogramas
plt.hist(muestras_unif, bins=bins, alpha=0.5, label='Uniforme')
plt.hist(muestras_norm, bins=bins, alpha=0.5, label='Normal')
plt.legend()
plt.show()



# Estimar la entropía de la uniforme cuando las distribución q
# que nos provee las muestras es una N(3,1)

muestras_normal = np.random.normal(3, 1, N)

# Estimar la entropía de la distribución uniforme usando las muestras de la normal
H_uniform_from_normal = H(muestras_normal, bins)
print("Entropía de la distribución uniforme (estimada desde una N(3,1)): ", H_uniform_from_normal)

# Visualizar el histograma de las muestras de la normal
plt.hist(muestras_normal, bins=bins, alpha=0.5, label='Normal(3,1)')
plt.legend()
plt.show()


# Se obtienen las siguientes muestras de dos normales, calcular la primer corrección de la 
# mezcla con el algoritmo EM. Con el algoritmo EM obtener la estimación de los
# parámetros de las distnbuciones.

#A MANO

