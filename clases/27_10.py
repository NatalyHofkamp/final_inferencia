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

def get_gauss(mu, sigma,i):
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    y = norm.pdf(x, mu, sigma)
    plot(x, y, label=f'Curva de campana ({i})')
    plt.legend()

def get_mu(muestra):
    u0 = muestra[0]
    sigma0 = 100
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

def test(muestra,u1,u2,v1,v2,clase):
    prob1 = norm.pdf(muestra,u1,v1)
    prob2 = norm.pdf(muestra,u2,v2)
    if clase == 1 and prob1>prob2:
        return True
    if clase == 2 and prob2>prob1:
        return True
    return False


def training(muestra,label):
    size = int(len(muestra)*0.7)
    mixed_muestra = np.random.permutation(muestra)
    training_muestra = mixed_muestra[:size]
    testing_muestra = mixed_muestra[size:]
    muuu = []
    varianzas = []
    for i in range(1,len(training_muestra)):
        var,mu= get_mu(training_muestra[:i])
        muuu.append(mu)
        varianzas.append(var)
    f_mu = muuu[-1]
    f_var = varianzas[-1]
    # get_gauss(f_mu,f_var,label)
    return f_mu,f_var,testing_muestra

def performance (muestra1,muestra2):
    u1,v1,testing_muestra1 = training(muestra1,1)
    u2,v2,testing_muestra2= training(muestra2,2)
    true_cases1 = 0
    for muestra in testing_muestra1:
        if test(muestra,u1,u2,v1,v2,1):
            true_cases1+=1
    true_cases2 = 0
    for muestra in testing_muestra2:
        if test(muestra,u1,u2,v1,v2,2):
            true_cases2+=1

    p1 = len(muestra1)/ (len(muestra1)+len(muestra2))
    # print(f'performance muestra 1{true_cases1/len(testing_muestra1)}')
    # print(f'performance muestra 2 {true_cases2/len(testing_muestra2)}')

    p2 = len(muestra2)/ (len(muestra1)+len(muestra2))
    perf = p1 * true_cases1/len(testing_muestra1) + p2 * true_cases2/len(testing_muestra2)
    return perf

def get_prom_perf(muestra1,muestra2):
    performances = []
    for _ in range(100):
        perfo = performance (muestra1,muestra2)
        performances.append(perfo)
    perfo_media = np.mean(performances)
    perfo_desv =(np.var(performances))**1/2
    return perfo_media,perfo_desv
    
def bootstraping (muestra1,muestra2):
    cachitos1 = [muestra1[i:i+22] for i in range(0, 220, 22)]
    cachitos2 = [muestra2[i:i+22_media = np.mean(performances)
    perfo_desv =(np.var(performances))**1/2
    return perfo_media,perfo_desv
    
def bootstraping (muestra1,muestra2):
    cachitos1 = [muestra1[i:i+22] for i in range(0, 220, 22)]
    cachitos2 = [muestra2[i:i+22] for i in range(0, 220, 22)]

    prom_perfo_med = []
    prom_perfo_desv = []

    for cacho1, cacho2 in zip(cachitos1, cachitos2):
        perfo_media, perfo_desv = get_prom_perf(cacho1, cacho2)
        prom_perfo_med.append(perfo_media)
        prom_perfo_desv.append(perfo_desv)

    print(f'media promedio {np.mean(prom_perfo_med)}')
    print(f'desvío estándar promedio {np.mean(prom_perfo_desv)}')
    print(f'error estándar promedio {np.mean(prom_perfo_desv)/np.sqrt(10)}')



def main():
    data = {'C1':[[],[]],'C2':[[],[]]} 
    charge_data(data,'datos.csv')
    print("-------ancho---------")
    bootstraping(data['C1'][1],data['C2'][1])
    print("-------largo---------")
    bootstraping(data['C1'][0],data['C2'][0])
    




    
if __name__ =='__main__':
    main()