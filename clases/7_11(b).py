from pylab import *
from scipy.stats import norm
def charge_data(filename):
    data_matrix = []
    with open(filename, 'r') as f:
        lineas = f.readlines()
        for l in lineas:
            l = l.strip().split(',')
            data_matrix.append([int(l[1]), int(l[2])])
    return np.array(data_matrix)


data = charge_data('datos.csv')
# Inicialización de parámetros
num_components = 2
n_iterations = 100
mu = np.random.rand(num_components) * data.max()  # Medias iniciales
sigma = np.random.rand(num_components) * data.std()  # Desviaciones estándar iniciales
phi = np.ones(num_components) / num_components  # Probabilidades iniciales de componentes

for iteration in range(n_iterations):
    # Paso E (Expectation): Calcular probabilidades de pertenencia a cada componente para cada dato
    responsibilities = np.zeros((len(data), num_components))
    for k in range(num_components):
        for i in range(len(data)):
            # Calcular la probabilidad para cada punto de datos y componente
            responsibilities[i, k] = phi[k] * norm.pdf(data[i], loc=mu[k], scale=sigma[k])
    responsibilities /= responsibilities.sum(axis=1)[:, np.newaxis]

    # Paso M (Maximization): Actualizar los parámetros del modelo
    for k in range(num_components):
        Nk = responsibilities[:, k].sum()
        mu[k] = np.sum(responsibilities[:, k] * data) / Nk
        sigma[k] = np.sqrt(np.sum(responsibilities[:, k] * (data - mu[k])**2) / Nk)
        phi[k] = Nk / len(data)


# Imprimir los resultados finales
print("Medias finales:", mu)
print("Desviaciones estándar finales:", sigma)
print("Probabilidades finales de componentes:", phi)