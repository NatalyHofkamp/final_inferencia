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
def get_conju_marg(ancho,largo):
    H, x_edges, y_edges = np.histogram2d(ancho,largo, bins=(5, 5), range=((min(ancho), max(ancho)),(min(largo), max(largo))))
    p_conju = H/np.sum(H)
    margi_A= np.sum(H, axis=0)
    p_A = margi_A/np.sum(H)
    margi_L = np.sum(H,axis = 1)
    p_L = margi_L/np.sum(H)
    return p_conju,p_A,p_L

def mean(data):
    return sum(data)/len(data)

def var(data):
    med = mean(data)
    values = [(v - med)**2 for v in data]
    return mean(values)


def likelihood(muestras):
    likeli= 1
    u = mean(muestras)
    sigma = np.sqrt(var(muestras))
    for m in muestras:
        likeli *= ((1/np.sqrt(2*np.pi)*sigma)*np.exp(-(m-u)**2/2*sigma**2))
    return likeli

def dist_uni(min, max, size):
    return np.random.uniform(low=min, high=max, size=size)


def dist_exp(lamb, x):
    return lamb*np.exp(-lamb*x)

def main():
    data = {'C1':[[],[]],'C2':[[],[]]} 
    charge_data(data,'TP3/datos.csv')
    conj1,mg_a1,mg_l1 = get_conju_marg(data['C1'][0],data['C1'][1])
    conj2,mg_a2,mg_l2 = get_conju_marg(data['C2'][0],data['C2'][1])
    print("likelihood a una normal del ancho clase 1→",likelihood(mg_a1))
    print("likelihood a una normal del largo clase 1→",likelihood(mg_l1))
    print("likelihood a una normal del ancho clase 2→",likelihood(mg_a2))
    print("likelihood a una normal del largo clase 2→",likelihood(mg_l2))
    #_------------------------------------------------------
    muestra = dist_uni(0,1,1000)
    lamb = 1
    muestra_expo = dist_exp(lamb,muestra)
    plt.scatter(muestra_expo,muestra)
    plt.show()
    values = []
    for lamb in range(0,6):
        muestra_expo = dist_exp(lamb,2)
        values.append(muestra_expo)
    plt.scatter(list(range(0,6)),values)
    plt.show()

    

if __name__ =='__main__':
    main()