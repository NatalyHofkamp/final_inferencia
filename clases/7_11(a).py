from pylab import *
from scipy.spatial.distance import pdist


def charge_data(filename):
    data_matrix = []
    with open(filename, 'r') as f:
        lineas = f.readlines()
        for l in lineas:
            l = l.strip().split(',')
            data_matrix.append([int(l[1]), int(l[2])])
    return np.array(data_matrix)


def get_avg_desv(matriz):
    return np.std(matriz)

def histograms (datos,d):
    distancias = pdist(datos)
    plt.hist(distancias, bins='auto')
    plt.xlabel('Distancia')
    plt.ylabel('Frecuencia')
    plt.title(d[1], fontsize = 20, fontweight= 3)
    plt.savefig('results/histogram_'+d[1]+'.jpg')
    plt.show()

def get_centroids(datos, k,avg_desv):
    centroids = datos[np.random.choice(len(datos), k, replace=False)]
    variances = []  

    i = 0
    while True:
        i += 1
        asignaciones = np.argmin(np.linalg.norm(datos[:, np.newaxis] - centroids, axis=-1), axis=-1)
        nuevos_centroids = np.array([datos[asignaciones == j].mean(axis=0) for j in range(k)])

        # Calcula y registra la varianza de cada cluster en cada iteración
        cluster1, cluster2 = clusters(datos, nuevos_centroids, avg_desv)
        variance_cluster1 = np.std(cluster1)
        variance_cluster2 = np.std(cluster2)
        avg_desv= variance_cluster1+variance_cluster2
        variances.append(variance_cluster1+variance_cluster2)

        if np.all(centroids == nuevos_centroids):
            break
        centroids = nuevos_centroids

    return centroids, variances


def clusters(datos, centroids,avg_desv):
    cluster1 = []
    cluster2 = []
    for i in datos:
        cal1 = np.exp(-((np.linalg.norm(i - centroids[0]))**2) / (2 * (avg_desv**2)))
        cal2 = np.exp(-((np.linalg.norm(i - centroids[1]))**2) / (2 * (avg_desv**2)))
        if cal1 > cal2:
            cluster1.append((i))
        else:
            cluster2.append((i))
    return cluster1, cluster2

def graph_cords(cord1,cord2,centroids):
    x_cord1,y_cord1= zip(*cord1)
    x_cord2,y_cord2= zip(*cord2)
    plt.xlabel('largo')
    plt.ylabel('ancho')
    plt.scatter(x_cord1,y_cord1, color='pink', label='Cluster 1')
    plt.scatter(x_cord2,y_cord2, color='lightblue', label='Cluster 2')
    plt.scatter(centroids[0, 0],centroids[0, 1], marker='x', color='red', label='Centroide 1')
    plt.scatter(centroids[1, 0], centroids[1, 1], marker='x', color='blue', label='Centroide 2')
    plt.legend()
    plt.savefig('clusters.jpg')
    plt.show()

def print_clusters(cluster1,cluster2,centroids):
    return graph_cords(cluster1,cluster2,centroids)



def main ():
    filename = "datos.csv"  
    mat = charge_data(filename)
    avg_desv = get_avg_desv(mat)
    # print('varianza→',avg_desv)
    k= 2
    centroids, variances = get_centroids(mat, k,avg_desv)
    
    # histograms(mat,2)
    cluster1,cluster2 = clusters(mat, centroids,avg_desv)
    print_clusters(cluster1,cluster2,centroids)
    plt.scatter(range(len(variances)),variances)
    plt.show()


if __name__ == '__main__':
    main() 

