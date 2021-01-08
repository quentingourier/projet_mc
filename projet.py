import numpy as np
from numpy.lib.shape_base import _replace_zero_by_x_arrays
from numpy.linalg.linalg import get_linalg_error_extobj
import matplotlib.pyplot as pp

#DEFINITIONS

def DecompositionGS(A):
    """ Calcul de la décomposition QR de A une matrice carrée.
    L'algorithme de Gram-Schmidt est utilisé.
    La fonction renvoit (Q,R) """
    n,m=A.shape
    Q=np.zeros((n,m))
    R=np.zeros((n,m))

    for j in range(m):
        for i in range(j):
            R[i,j]=Q[:,i]@A[:,j]
        w=A[:,j]
        for k in range(j):
            w=w-R[k,j]*Q[:,k]
        norme=np.linalg.norm(w)
        if norme ==0:
            raise Exception('Matrice non inversible')
        R[j,j]=norme
        Q[:,j]=w/norme
    R = R[~np.all(R == 0, axis=1)]
    return Q,R

def ResolTriSup(T,b):
    """Résolution d'un système triangulaire supérieur carré
    Tx=b
    La fonction ne vérifie pas la cohérence des dimensions de T et b
    ni que T est triangulaire supérieure.
    La fonction rend x sous la forme du même format que b."""
    n,m=T.shape
    x=np.zeros(b.shape)
    for i in range(n-1,-1,-1):
        S=T[i,i+1:]@x[i+1:]
        x[i]=(b[i]-S)/T[i,i]
    return x

def ResolTriInf(T,b):
    """Résolution d'un système triangulaire inférieur carré
    Tx=b
    La fonction ne vérifie pas la cohérence des dimensions de T et b
    ni que T est triangulaire inférieure.
    La fonction rend x sous la forme du même format que b."""
    n,m=T.shape
    x=np.zeros(n)
    for i in range(n):
        S=T[i,:i]@x[:i]
        x[i]=(b[i]-S)/T[i,i]
    x=np.reshape(x,b.shape)
    return x

def Cholesky(A):
    """
    Fonction qui calcule L la matrice de la décomposition de
    Cholesky de A une matrice réelle symétrique définie positive
    (A=LL^T où L est triangulaire inférieure).
    La fonction ne vérifie pas que A est symétrique.
    La fonction rend L.
    """
    n,m=A.shape
    if n != m:
        raise Exception('Matrice non carrée')
    L=np.zeros((n,n))
    for i in range(n):
        s=0.
        for j in range(i):
            s=s+L[i,j]**2
        R=A[i,i]-s
        if R<=0:
            raise Exception('Matrice non définie positive')
        L[i,i]=np.sqrt(R)
        for j in range(i+1,n):
            s=0.
            for k in range(i):
                s=s+L[i,k]*L[j,k]
            L[j,i]=(A[j,i]-s)/L[i,i]
    return L, L.T


def donnees_partie3():
    """ Fonction qui donne les données à traiter dans la partie 3
    du projet. 
    ----------------
    Utilisation : x,y=donnes_partie3()
    """
    x=np.array([ 3.58, -2.26,  1.17,  7.09,  1.3 , -4.82, -4.83,  1.53,  5.73,
       -3.44,  4.04,  2.99,  3.59, -4.66, -0.61,  0.67, -4.02, -1.91,
        6.58, -5.07,  5.18, -1.67, -2.6 ,  4.27,  4.  , -5.36, -2.1 ,
        5.94, -3.92, -3.29,  6.39,  2.04, -4.66,  7.73, -4.26,  4.26,
       -4.15, -4.67, -0.73, -4.8 ,  5.15, -2.9 ,  6.55,  5.7 ,  6.15,
        5.46,  0.1 , -2.46, -4.52,  7.01,  6.79, -0.04,  7.25, -2.01,
        7.07, -2.02, -4.57,  3.11,  1.01,  6.38, -4.69,  7.19, -4.22,
       -5.08,  6.9 ,  4.28, -3.31,  6.58, -1.71,  6.28, -3.9 ,  6.88,
       -3.76,  4.53,  6.31,  6.54,  7.17,  7.3 ,  6.38, -1.17, -0.22,
       -0.64,  3.91,  2.11,  1.66, -1.66, -4.1 ,  6.16,  7.54, -1.44,
        5.57,  4.85,  7.04, -4.64,  6.67, -4.93,  6.92, -3.11,  0.17,
        3.95])

    y=np.array([-6.71,  4.17, -6.94,  0.19, -6.9 , -2.03, -0.52,  5.14, -4.87,
       -4.94,  4.12,  4.99,  4.83,  0.78, -6.75, -7.3 , -4.05,  4.58,
       -3.26, -0.12,  4.38,  4.14, -5.99,  4.38, -6.41,  0.03, -6.2 ,
        3.04,  2.  ,  3.26,  2.53,  4.94, -2.35,  0.85, -4.18,  4.79,
        1.99, -2.74, -6.75, -2.97,  3.87, -5.26, -4.45,  3.87,  2.74,
       -5.24, -6.97,  4.07, -2.31, -0.11,  1.71, -7.01, -0.62, -5.79,
        0.82, -5.86,  1.87,  5.09, -7.09, -3.97, -2.9 , -2.3 , -4.63,
       -1.17,  0.86,  4.68,  3.23, -3.48, -6.54, -4.76,  2.55,  0.92,
        3.19,  3.97,  2.93, -3.28,  0.26, -2.52, -4.63, -6.4 ,  5.05,
       -6.75, -6.41,  5.18,  5.49,  4.13,  2.57,  2.58,  0.04,  4.8 ,
        3.59,  4.09,  1.15, -2.68,  1.24, -0.9 , -3.56,  3.29, -7.06,
        4.46])

    return x,y

def res(Aaug):
    n, m = np.shape(Aaug)
    X = np.zeros(n)
    for i in range(n-1, -1, -1): 
        somme = 0
        for k in range(i,n):
            somme = somme + X[k]*Aaug[i,k]
        X[i] = (Aaug[i, n] - somme) / Aaug[i, i]
    return X

def ResolMCEN(A, b):
    #AT*Ax = AT*b où AT*A est A_EN pour A_EquationsNormales
    #et où AT*b est B_EN pour B_EquationsNormales
    A_EN = np.dot(A.T, A)#détermination du système d'équations normales
    B_EN = np.dot(A.T, b)
    L, U = Cholesky(A_EN)#resolution par cholesky
    T = np.concatenate((L, B_EN), axis = 1)
    Y = ResolTriInf(T,B_EN)
    Aaug = np.concatenate((U, Y), axis = 1)
    x = res(Aaug)
    return x 

def ResolMCQR(A, b):
    Q, R = DecompositionGS(A)
    B_EN = np.dot(Q.T, b)
    x=np.squeeze(np.asarray(ResolTriSup(R, B_EN)))
    return x

def ResolMCNP(A, b):
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond = -1)
    print(x, residuals, rank, s)
    return np.squeeze(np.asarray(x))

#PROGRAMME

#1.1 Avec les equations normales
A = np.array([[1, 2], [2, 3], [-1, 2]])
b = np.array([[12], [17], [6]])
x = ResolMCEN(A, b)

#1.2 Avec la decomposition QR
A = np.array([[1, 21], [-1, -5], [1, 17], [1, 17]])
b = np.array([3, -1, 1, 1])
x = ResolMCQR(A, b)

#1.3 La fonction de numpy
A = np.array([[1, 21], [-1, -5], [1, 17], [1, 17]])
b = np.array([3, -1, 1, 1])
x = ResolMCNP(A, b)


#2.1 Comparaison des solutions

#Tests sur EX1
A = np.array([[1, 2], [2, 3], [-1, 2]])
b = np.reshape(np.array([12, 17, 6]), (3,1))
print("\nEX1 :\n")
print("Par la méthode des équations normales :", ResolMCEN(A, b))
print("Par la méthode de la décomposition QR :", ResolMCQR(A, b))
print("Par la méthode de la fonction numpy   :", ResolMCNP(A, b))

#Tests sur EX2
A = np.array([[1, 21], [-1, -5], [1, 17], [1, 17]])
b = np.reshape(np.array([3, -1, 1, 1]), (4,1))
print("\nEX2 :\n")
print("Par la méthode des équations normales :", ResolMCEN(A, b))
print("Par la méthode de la décomposition QR :", ResolMCQR(A, b))
print("Par la méthode de la fonction numpy   :", ResolMCNP(A, b))

#Tests sur EX3
#creation de A et b
xi = [0.3, -2.7, -1.9, 1.2, -2.6, 2.7, 2.0, -1.6, -0.5, -2.4]
yi = [2.8, -9.4, -4.5, 3.8, -8.0, 3.0, 3.9, -3.5, 1.3, -7.6]
A = np.zeros((len(xi), 3))
b = np.zeros((len(yi), 1))
for i in range(len(xi)):
    A[i, 0] = 1
    A[i, A.shape[1]-2] = xi[i]
    A[i, A.shape[1]-1] = xi[i]**2
    b[i,:] = yi[i]
print("\nEX3 :\n")
print("Par la méthode des équations normales :", ResolMCEN(A, b))
print("Par la méthode de la décomposition QR :", ResolMCQR(A, b))
print("Par la méthode de la fonction numpy   :", ResolMCNP(A, b))


#2.2 Vérification du minimum
#On se place dans le cadre de l'exercice 2
A = np.array([[1, 21], [-1, -5], [1, 17], [1, 17]])
b = np.array([3, -1, 1, 1])
x_ = np.reshape(ResolMCQR(A, b), (2,))
for i in range(10**6):
    x = np.zeros((2,))
    while np.linalg.norm(x-x_) >= 10**-3:
        x = x_ + np.array((10**-5)*(np.random.choice((-1, 1)))*np.random.rand(2,))

    erreur_x = np.linalg.norm(np.dot(A, x) - b)
    erreur_x_ = np.linalg.norm(np.dot(A, x_) - b)
    if  erreur_x >= erreur_x_:
        print("\n||Ax-b|| >= ||Ax*-b|| est vérifié")
    else:
        print("||Ax-b|| >= ||Ax*-b|| non vérifié")
    print("\n||Ax-b|| = ", erreur_x)
    print("||Ax*-b|| = ", erreur_x_)     

#3.1 À la main (FAIT)
#3.2 À la main (FAIT)

#3.3 Le cercle le plus fidèle
x, y = donnees_partie3()
s_n = len(x)
#somme des xi
s_xi, s_xi2, s_xi3 = sum(x), sum(x**2), sum(x**3)
#somme des produits xi*yi
s_xiyi, s_xi2yi, s_xiyi2 = sum(x*y), sum((x**2)*y), sum(x*(y**2))
#somme des yi
s_yi, s_yi2, s_yi3 = sum(y), sum(y**2), sum(y**3)
#calcul du centre et du rayon par la méthode des MC
A = np.array([[s_xi, s_yi, s_n], [s_xi2, s_xiyi, s_xi], [s_xiyi, s_yi2, s_yi]])
b = np.array([s_xi2+s_yi2, s_xi3+s_xiyi2, s_xi2yi+s_yi3])
solution = ResolMCNP(A, b)
coeff = np.array([solution[0]/2, solution[1]/2, solution[2]])
rayon = (x[2]**2+y[2]**2-2*x[2]*coeff[0]-2*y[2]*coeff[1]+coeff[0]**2+coeff[1]**2)**0.5
centre = (coeff[0], coeff[1])

#3.4 Représentation graphique
#nuage de points
pp.scatter(x, y, label = 'points')
pp.scatter(coeff[0], coeff[1], label = 'centre', color = 'red')
#cercle calculé
cercle = pp.Circle((coeff[0], coeff[1]), rayon, color = 'r', fill = False)
pp.gca().add_artist(cercle)
pp.xlim(-8, 9)
pp.xlabel('x')
pp.ylabel('y')
pp.title("Cercle calculé à l'aide du problème des moindres carrés")
pp.legend()
pp.show()
