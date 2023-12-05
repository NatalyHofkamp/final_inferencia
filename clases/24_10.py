from pylab import *
from scipy.stats import expon,norm,poisson,chi2
def charge_data(cant_dat,filename):
    with open (filename,'r') as f:
        lineas = f.readlines()
        for l in lineas:
            l= l.strip().split(',')
            for data_type in cant_dat:
                if l[0] == data_type:
                    cant_dat[data_type][0].append(int(l[1]))
                    cant_dat[data_type][1].append(int(l[2]))
def mean(data):
    return sum(data)/len(data)

def var(data):
    med = mean(data)
    values = [(v - med)**2 for v in data]
    return mean(values)


def get_gauss(mu, sigma,i):
    # Valores para el eje x
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)


    # Calcula la densidad de probabilidad de la distribución normal en esos valores
    y = norm.pdf(x, mu, sigma)

    # Crea el gráfico de la curva de la campana en el eje proporcionado
    plot(x, y, label=f'Curva de campana ({i})')
    plt.legend()

def get_mu(muestra):
    u0 = muestra[0]
    sigma0 = 50
    n = len (muestra)
    sigma = np.var(muestra)
    prom = get_prom(muestra)
    varianza = get_var(sigma0,sigma,n)
    mu = (n*sigma0**2/(n*sigma0**2+sigma**2))*prom + ((sigma**2)/(n*sigma0**2+sigma**2))*u0
    return varianza,mu

def get_var(sigma0,sigma,n):
    return ((sigma*sigma0)**2/(n*sigma0**2+sigma**2))

def get_prom (muestras):
    return np.sum(muestras)/len(muestras)

def main():
    data = {'C1':[[],[]],'C2':[[],[]]} 
    charge_data(data,'datos.csv')
    muestra = data['C1'][0]
    muuu = []
    varianzas = []
    mu_real = np.mean(muestra)
    for i in range(1,len(muestra)):
        var,mu= get_mu(muestra[:i])
        muuu.append(mu)
        varianzas.append(var)
    # Corrección: Debes convertir mu_real a una lista para poder graficarlo.
    plt.scatter(list(range(1, len(muestra))), muuu, label="Estimación de mu")
    # plt.plot(list(range(1, len(muestra)), ([mu_real] * (len(muestra) - 1)), label="Mu real", linestyle='--'))
    # Agregar una línea horizontal en el eje x
    plt.axhline(y=mu_real, color='red', linestyle='--', label="Línea en x")

    # Corrección: Agregar etiquetas y leyenda al gráfico.
    plt.xlabel("Tamaño de la muestra")
    plt.ylabel("Valor de mu")
    plt.legend()
    plt.figure()
    plt.scatter(list(range(1, len(muestra))), varianzas, label="Estimación de varianza")
    var_real = np.var(muestra)
    plt.axhline(y=var_real, color='red', linestyle='--', label="Línea en x")
    plt.show()

    for i in range(1,20,2):
        mu = muuu[i]
        print(mu)
        sigma = varianzas[i]
        get_gauss(mu, sigma,i)
if __name__ =='__main__':
    main()