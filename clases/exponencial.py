from pylab import *
from scipy.stats import expon,norm,poisson,chi2
def dist_uni(min, max, size):
    return np.random.uniform(low=min, high=max, size=size)


def dist_exp(lamb, x):
    return lamb*np.exp(-lamb*x)

def muestra_exp(lamb, size):
    return expon.rvs(scale = 1/lamb, size = size)

def main():
    muestra = np.linspace(0,10,100)
    lamb = 1
    muestra_expo = dist_exp(lamb,muestra)
    plt.scatter(muestra,muestra_expo)
    plt.show()
    #-------------------------------------
    values = []
    for lamb in range(0,6):
        muestra_expo = dist_exp(lamb,2)
        values.append(muestra_expo)
    plt.scatter(list(range(0,6)),values)
    plt.show()
    #-------------------------------------
    thetas = []
    sizes = list(range(50,100000,50))
    for size in sizes:
        muestra_expo = np.random.exponential(3,size)
        suma = np.sum(muestra_expo)
        n= len(muestra_expo)
        theta_esp = n/suma
        thetas.append(theta_esp)

    plt.plot(sizes,thetas)
    plt.show()
#--------------------------------------------------

# 2. Let x have a uniform density
# 0 ≤ x ≤ θ
# otherwise.
# 1/θ
# 0
# p(x|θ) ∼ U (0, θ) =
# (a) Suppose that n samples D = {x 1 , ..., x n } are drawn independently according to
# p(x|θ). Show that the maximum likelihood estimate for θ is max[D], i.e., the
# value of the maximum element in D.
# (b) Suppose that n = 5 points are drawn from the distribution and the maximum
    # value of which happens to be max x k = 0.6. Plot the likelihood p(D|θ) in the
    # k
    # range 0 ≤ θ ≤ 1. Explain in words why you do not need to know the values of
    # the other four points.
    muestra_uni = dist_uni(0,1,5)
    #el theta de la distirbución es el máximo de las muestras que tenemos
    #si lo ploteamos y el likelihood es 0.6, no nos improta saber qué muestras tiene
    #Porque sabemos que la última es la que tiene el máximo PARA LOS PUNTOS ANTEIRORES
    #la probabilidad es la misma que para theta
    #P(x1<0.6) = 1/0.6
    #p(X1 , .. , XN|THETA)=(1/0.6)**5



    

if __name__ =='__main__':
    main()