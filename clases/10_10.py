from pylab import *
from scipy.stats import pearsonr

# # Generar las distribuciones normales a, b y c
# a = np.random.normal(loc=0, scale=1, size=1000)
# b = np.random.normal(loc=0, scale=1, size=1000) 
# # Calcular la correlación de Pearson entre a y b antes de agregar c
# correlation_coefficient_before = pearsonr(a, b)
# print(f"Coeficiente de correlación de Pearson (antes de agregar c): {correlation_coefficient_before[0]}")
# a += c
# b += c
# # Calcular la correlación de Pearson entre a y b después de agregar c
# correlation_coefficient_after = pearsonr(a, b)
# print(f"Coeficiente de correlación de Pearson (después de agregar c): {correlation_coefficient_after[0]}")

c_values = [10,15,20,30,50,100,150,200]
correlaciones = []
a = np.random.normal(loc=0, scale=1, size=1000)
b = np.random.normal(loc=0, scale=1, size=1000) 
for x in c_values:
    c = x* np.random.normal(loc=0, scale=1, size=1000)
    new_a = a + c
    new_b = b + c
    # Calcular la correlación de Pearson entre a y b después de agregar c
    correlaciones.append(pearsonr(new_a, new_b)[0])

plt.scatter(correlaciones,c_values)
plt.show()

def entropia_conjunta(p):
    h = 0
    for x in p :
        for y in x:
            if y != 0:
              h-= y*np.log2(y)
    return h

def demo_indep(p_conju,p_A,p_L):
    e_conju = entropia_conjunta(p_conju)
    entro_A= entropia(p_A)
    entro_L = entropia(p_L)
    x = entro_A + entro_L
    if abs(e_conju - x )<= 0.1:
        return 'son independientes'
    else: 
        return 'no son independientes'

print(f'se demuestra que el alto y ancho de la clase 1 {demo_indep(conj1,p_a1,p_l1)}')
print(f'se demuestra que los anchos de ambas clases  {demo_indep(conj1_2A,p_A1,p_A2)}')