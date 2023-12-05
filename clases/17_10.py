from pylab import *

def f(x):
    return -np.log2(x)

def histo_densidad (muestra,bins):
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

if __name__ == '__main__':
    main()