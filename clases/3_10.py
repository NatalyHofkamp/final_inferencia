from pylab import *
# from data_reader import *
# #---------------ENTROPIA -------------------
def charge_data(cant_dat,filename):
    with open (filename,'r') as f:
        lineas = f.readlines()
        for l in lineas:
            l= l.strip().split(',')
            for data_type in cant_dat:
                if l[0] == data_type:
                    cant_dat[data_type][0].append(int(l[1]))
                    cant_dat[data_type][1].append(int(l[2]))
def H (p):
    return -p*np.log2(p)-(1-p)*log2(1-p)

def entropia_bernoulli():
    Hs = []
    ps = np.linspace(0,1,1000)
    for p in ps:
        if p != 0 and p != 1:
            Hs.append(H(p))
        else: 
            Hs.append(0)

    plt.plot(ps,Hs)
    plt.show()

#--------------CON LAS HOJAS / entropía conjunta ----------

def charge(data,clase1,clase2,filename ):
    with open(filename, 'r') as f:
        lineas = f.readlines()
        for l in lineas:
            l = l.strip().split(',')
            data_type = l[0]
            if data_type in data:
                data[data_type].append((int(l[1]), int(l[2])))
                if int(data_type) == 1:
                    clase1[0].append(int(l[1]))
                    clase1[1].append(int(l[2]))
                else:
                    clase2[0].append(int(l[1]))
                    clase2[1].append(int(l[2]))

def conjunta(dist):
    conj = {}
    for tupla in dist:
        if tupla in conj:
            conj[tupla] += 1
        else:
            conj[tupla] = 1
    return conj


def entropia_conjunta(ancho,largo):
    h = 0
    H, x_edges, y_edges = np.histogram2d(ancho,largo, bins=(5, 5), range=((min(ancho), max(ancho)),(min(largo), max(largo))))
    p = H / np.sum(H)  # Normaliza el histograma para obtener la distribución de probabilidad conjunta
    for x in p :
        for y in x:
            if y != 0:
              h+= entropia(y)
    print('conjunta→',h)
    return h


#-----------------ENTROPIA CONDICIONAL----------------

def entropia_condicional (ancho,largo):

    H, x_edges, y_edges = np.histogram2d(ancho,largo, bins=(5, 5), range=((min(ancho), max(ancho)),(min(largo), max(largo))))
    p = H / np.sum(H)  # Normaliza el histograma para obtener la distribución de probabilidad conjunta
    # Calcula la probabilidad de A(P(A)) → esto esta bueno para la demo_conju
    p_A = np.sum(H, axis=0)
    # Calcula la probabilidad conjunta de L y A (ancho y largo)
    p_conju= H
    # Calcula la probabilidad condicional de L(largo) dado A(ancho)
    p_condi= p_conju/ p_A
    h= 0
    for x in range(len(p_condi)):
        for i in range(len(p_condi)):
                if p[x,i] != 0 and p_condi[x,i] !=0:
                    h-= p[x,i]*np.log2(p_condi[x,i])
    print ("entropia condicional→",h)
    return h

#teoría → demo_conju de que la encontropía de la marginal + la condicional==  a la entropía conjunta
#parece importante→ condicional baja la entropía siempre y cuando las variables sean independientes
#esto demuestra que la conjunta > marginal 
def entropia (p):
     return -p*np.log2(p)
def entropia_marginal (p_m):
    entro_muestra= 0
    for x in p_m:
        if x != 0:
            entro_muestra += entropia(x)
    return entro_muestra

def demo_conju(ancho,largo):
    H, x_edges, y_edges = np.histogram2d(ancho,largo, bins=(5, 5), range=((min(ancho), max(ancho)),(min(largo), max(largo))))
    p = H / np.sum(H)  # Normaliza el histograma para obtener la distribución de probabilidad conjunta
    # Calcula la probabilidad de A(P(A)) → esto esta bueno para la demo_conju
    marginal = np.sum(H, axis=0)
    p_m = marginal/np.sum(marginal)
    #si no tuvieras que calcular la conjunta, y necesitas una marginal, podes usar esto
        # H_m, x_edges = np.histogram(largo, bins=5, range=(min(largo), max(largo)))
        # p_m= H_m / np.sum(H_m)  # Normaliza el histograma para obtener la distribución de probabilidad
    # # Calcula la entropía de la variable "ancho"
    entro_ancho= entropia_marginal(p_m)
    entro_conjunta = entropia_conjunta(ancho,largo)
    entro_condicional = entropia_condicional(ancho,largo)
    if abs(entro_conjunta - (entro_ancho+entro_condicional)) <= 0.001:
        print('sip')
    else: 
        print('nop')

def gen_conju_marg(ancho,largo):
    H, x_edges, y_edges = np.histogram2d(ancho,largo, bins=(5, 5), range=((min(ancho), max(ancho)),(min(largo), max(largo))))
    p_conju = H/np.sum(H)
    margi_A= np.sum(H, axis=0)
    p_A = margi_A/np.sum(H)
    margi_L = np.sum(H,axis = 1)
    p_L = margi_L/np.sum(H)
    return p_conju,p_A,p_L


def demo_indep(ancho,largo):
    h = 0
    H, x_edges, y_edges = np.histogram2d(ancho,largo, bins=(5, 5), range=((min(ancho), max(ancho)),(min(largo), max(largo))))
    e_conju = entropia_conjunta(ancho,largo)
    margi_A= np.sum(H, axis=0)
    p_A = margi_A/np.sum(margi_A)
    entro_A= entropia_marginal(p_A)
    margi_L = np.sum(H,axis = 1)
    p_L = margi_L/np.sum(margi_L)
    entro_L = entropia_marginal(p_L)
    x = entro_A + entro_L
    if abs(e_conju - (x) )<= 0.1:
        print('sip')
    else: 
        print('nop')

def relative_entropy(p, q):
    return p * np.log2(p / q)

def mutual_info(ancho,largo):
    H, x_edges, y_edges = np.histogram2d(ancho,largo, bins=(5, 5), range=((min(ancho), max(ancho)),(min(largo), max(largo))))
    p_conju = H/np.sum(H)
    margi_A= np.sum(H, axis=0)
    p_A = margi_A/np.sum(H)
    margi_L = np.sum(H,axis = 1)
    p_L = margi_L/np.sum(H)
    suma = 0
    for x in range(5):
        for y in range(5):
            if p_conju[x,y] >0: 
                suma +=  relative_entropy (p_conju[x,y],(p_A[y]*p_L[x]))
    print(suma)



def main():
    # entropia_bernoulli()
    data = {'C1':[[],[]],'C2':[[],[]]} 
    charge_data(data,'TP2/datos.csv')
    entropia_conjunta (data['C1'][0],data['C1'][1])
    entropia_conjunta (data['C2'][0],data['C2'][1])
    
    entropia_condicional(data['C1'][0],data['C1'][1])

    # print('ancho y largo C1 [conjunta mayor]:')
    demo_conju(data['C1'][0],data['C1'][1])
    # print('ancho y largo [independientes]:')
    # demo_indep(data['C1'][0],data['C1'][1])
    print('mutual infor-entropia relativa:')
    mutual_info(data['C1'][0],data['C1'][1])
    min_length = min(len(data['C1'][0]), len(data['C2'][0]))
    data['C1'][0] = data['C1'][0][:min_length]
    data['C2'][0] = data['C2'][0][:min_length]
    print('mutual infor-entropia relativa:')
    mutual_info(data['C1'][0],data['C2'][0])
    data['C1'][1] = data['C1'][1][:min_length]
    data['C2'][1] = data['C2'][1][:min_length]
    print('mutual infor-entropia relativa:')
    mutual_info(data['C1'][1],data['C2'][1])


    # #ancho de dos clases 
    # print('ancho dos clases [conjunta mayor]:')
    print('a')
    demo_conju(data['C1'][0],data['C2'][0])
    # print('ancho dos clases [independientes]:')
    # demo_indep(data['C1'][0],data['C2'][0])
    
if __name__ == '__main__':
    main()
