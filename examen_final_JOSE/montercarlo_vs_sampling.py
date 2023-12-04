import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Calcular la probabilidad de que la variable aleatoria con distribución N(0,1) sea mayor a 4
# 1. utilizando el método de integración de Montecarlo 
# genero una distribucion uniforme que va de -10 a 10 en x.
# y creo una uniforme que va de 0 al pico de la gausina que es 1/sqrt(2*pi) 
# despues cuento cuantos puntos de la uniforme caen mas alla de 4 y cuanto cunatos de la otra uniforme caen dentro de la otra normal 
# el area segun montecarlo
# despues pensar de 4 para adelate como una exponencial y calcular 
# la esperanza de p(x) > 4 mirandola desde la exponencial y
#  multiplicada por p(x)/q(x). p(x) es la normal y q(x) es la exponencial
#E[p(x>4)*(p(x)/q(x))]
# 2. utilizando importance sampling 
# 3 comparar los errores en la estimación


# Función de densidad de probabilidad de la normal estándar
def pdf_normal(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2*np.pi)

# Función de densidad de probabilidad de la distribución exponencial
def pdf_exponencial(x):
    return np.exp(-(x - 4))  

# Método de integración de Montecarlo
def montecarlo_probabilidad_mayor_4(num_puntos):
    x_uniforme = np.random.uniform(-10, 10, num_puntos)
    y_uniforme = np.random.uniform(0, 1/np.sqrt(2*np.pi), num_puntos)
    puntos_bajo_curva_normal = np.sum(y_uniforme < pdf_normal(x_uniforme))
    probabilidad_mayor_4 = puntos_bajo_curva_normal / num_puntos
    return probabilidad_mayor_4, x_uniforme[x_uniforme > 4]

# Método de importance sampling
def importance_sampling_probabilidad_mayor_4(num_puntos):
    muestras_q = np.random.exponential(size=num_puntos)
    pesos = pdf_normal(muestras_q) / pdf_exponencial(muestras_q)
    estimacion_importance_sampling = np.mean(pesos)
    return estimacion_importance_sampling, muestras_q[muestras_q > 4]

# Configuración
num_puntos = 100000

# Calcular probabilidad utilizando Montecarlo y obtener puntos
probabilidad_montecarlo, puntos_montecarlo = montecarlo_probabilidad_mayor_4(num_puntos)
print("Probabilidad utilizando Montecarlo:", probabilidad_montecarlo)

# Calcular probabilidad utilizando Importance Sampling y obtener puntos
probabilidad_importance_sampling, puntos_importance_sampling = importance_sampling_probabilidad_mayor_4(num_puntos)
print("Probabilidad utilizando Importance Sampling:", probabilidad_importance_sampling)

# Comparar errores en la estimación
error_relativo = np.abs(probabilidad_montecarlo - probabilidad_importance_sampling) / np.abs(probabilidad_montecarlo)
print("Error relativo en la estimación:", error_relativo, '\n')


# Función para graficar
def graficar():
    # Crear valores para las funciones de densidad de probabilidad
    x_vals_normal = np.linspace(-10, 10, 1000)
    x_vals_exponencial = np.linspace(4, 10, 1000)  # Ajustar el rango de valores para la exponencial
    y_normal = pdf_normal(x_vals_normal)
    y_exponencial = pdf_exponencial(x_vals_exponencial)

    # Crear el gráfico
    plt.figure(figsize=(10, 6))

    # Graficar las funciones
    plt.plot(x_vals_normal, y_normal, label='Normal', color='blue')
    plt.plot(x_vals_exponencial, y_exponencial, label='Exponencial', color='green')

    # Graficar todos los puntos generados en la simulación Montecarlo y Importance Sampling
    plt.scatter(puntos_montecarlo, np.zeros_like(puntos_montecarlo), color='red', marker='x', label='Puntos Montecarlo')
    plt.scatter(puntos_importance_sampling, np.zeros_like(puntos_importance_sampling), color='grey', marker='o', label='Puntos Importance Sampling')


    # Añadir línea vertical en x=4
    plt.axvline(x=4, color='purple', linestyle='--', label='x = 4')

    # Añadir leyenda y etiquetas
    plt.legend()
    plt.title('Funciones de Densidad y Puntos')
    plt.xlabel('x')
    plt.ylabel('Densidad de Probabilidad')

    plt.ylim(0, 0.7)

    # Mostrar el gráfico
    plt.show()

graficar()

# Función de densidad de probabilidad conjunta
def pdf_conjunta(x, y, pdf_X, pdf_Y):
    return pdf_X(x) * pdf_Y(y)

# Entropía Conjunta
def entropia_conjunta(x_vals, y_vals, pdf_X, pdf_Y):
    joint_entropy = -np.sum([pdf_conjunta(x, y, pdf_X, pdf_Y) * np.log(pdf_conjunta(x, y, pdf_X, pdf_Y)) for x in x_vals for y in y_vals])
    return joint_entropy

# Entropía Relativa
def entropia_relativa(x_vals, pdf_X, pdf_Q):
    relative_entropy = np.sum([pdf_X(x) * np.log(pdf_X(x) / pdf_Q(x)) for x in x_vals])
    return relative_entropy

# Valores para calcular la entropía
x_vals_normal = np.linspace(-10, 10, 1000)
y_vals_exponencial = np.linspace(4, 10, 1000)  # Ajustar el rango de valores para la exponencial

# Calcular entropía conjunta
entropy_joint = entropia_conjunta(x_vals_normal, y_vals_exponencial, pdf_normal, pdf_exponencial)
print("Entropía Conjunta:", entropy_joint)

# Calcular entropía relativa
entropy_relative = entropia_relativa(x_vals_normal, pdf_normal, pdf_exponencial)
print("Entropía Relativa:", entropy_relative)

