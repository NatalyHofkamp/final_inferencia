import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def calcular_densidad_probabilidad(x, mu, varianza):
    """
    Calcula la densidad de probabilidad gaussiana para un valor específico.

    Parámetros:
    - x (float): Valor para el cual calcular la densidad de probabilidad.
    - mu (float): Media de la distribución normal.
    - varianza (float): Varianza de la distribución normal.

    Retorna:
    - float: Densidad de probabilidad para el valor x.
    """
    desviacion_estandar = np.sqrt(varianza)
    densidad = norm.pdf(x, loc=mu, scale=desviacion_estandar)
    return densidad

# Ejemplo de uso
x_valor = 5.06
mu_normal = 4
varianza_normal = 1

# Calcular densidad de probabilidad
densidad_probabilidad = calcular_densidad_probabilidad(x_valor, mu_normal, varianza_normal)

# Imprimir resultado
print(f"La densidad de probabilidad para x={x_valor} con mu={mu_normal} y varianza={varianza_normal} es: {densidad_probabilidad}")

# Graficar la distribución normal
x = np.linspace(-5, 5, 1000)
# y = norm.pdf(x, loc=mu_normal, scale=np.sqrt(varianza_normal))

# plt.plot(x, y, label=f'N({mu_normal}, {varianza_normal})')
# plt.scatter(x_valor, densidad_probabilidad, color='red', label=f'Densidad para x={x_valor}')
# plt.title('Distribución Normal y Densidad de Probabilidad')
# plt.xlabel('x')
# plt.ylabel('Densidad de Probabilidad')
# plt.legend()
# plt.show()
