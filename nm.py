import matplotlib.pyplot as plt
from mpmath import *
mp.dps = 100
print(pi)

def newtonHP(f,df,x0,precision):
    n = 0;
    x = [x0];
    while (abs(f(x[n]))>10**(-precision)) and (n<100):
        n = n + 1;
        x.append(x[n-1]-f(x[n-1])/df(x[n-1]))
    return x

# root of a function by bisection
def bisection(f,a,b):
    for i in range(1000):
        c = (a+b)/2
        if f(a)*f(c) < 0:
            b = c
        else:
            a = c
    return c

def secant(f,x1,x2):
    a = [x1,x2]
    for i in range(16):
        a.append((a[-1]*f(a[-2])-a[-2]*f(a[-1]))/( f(a[-2])-f(a[-1])))
    return a

# lagragian
def pi(x,i,xs):
    xi = xs[i]
    xs.remove(xi)
    res = 1;
    for xj in xs:
        res *= (x-xj)/(xi-xj)
    return res

def lagrange(x,xs,ys):
    res = 0
    for i in range(len(ys)):
        res += ys[i]*pi(x,i,xs[:])
    return res

xs=[1,2,3,4,5,6];
ys=[0.,0.841471,0.909297,0.14112,-0.756802,-0.958924];
print(lagrange(1.5,xs,ys))

# polynomial fit
def findfit(m,xs,ys):
    A = np.zeros((m+1,m+1))
    b = np.zeros(m+1)
    for k in range(m+1):
        b[k] = sum(ys*xs**k)
        for j in range(m+1):
            A[k,j] = sum(xs**(k+j))
    sol = np.linalg.solve(A,b)
    def fit(x):
        a = 0
        for i in range(m+1):
            a += sol[i]*x**i
        return a
    return fit

import numpy as np

xs = np.array ([1 ,2 ,3 ,4 ,5 ,6])
ys = np.array ([-5.21659,2.53152,2.05687,14.1135,20.9673,33.5652])

ft = findfit(2,xs,ys)

# trapezoidal
def trapezoidal(f,a,b,n):
    fs = list(map(lambda x: f(x), np.linspace(a,b,n+1)));
    return ( sum(fs)-(fs[0]+fs[-1])*0.5)*(b-a)/n

# simpson
def simpson(f,a,b,n):
    r = np.linspace(a,b,n+1)
    total = 0
    for i in range(len(r)-1):
        total += (((b-a)/n)/6)*(f(r[i])+4*f((r[i]+r[i+1])/2)+f(r[i+1]))
    return total

def f(x):
    return 2/(x**2+1)

print(trapezoidal(f,-1.0,1.0,10.0))
print(simpson(f,-1.0,1.0,10.0))

p = 4 # run through powers and check numbers are the same
for i in range(1,20): # checking it experimentally
    print((simpson(np.cos,0.0,1.0,i)-np.sin(1.0))*(i**p))

# Gaussian Integration
import math
from scipy import integrate

a,b = -1,+1
def w(x):
    return math.exp(-x**2)

def h(k):
    return integrate.quad(lambda x: w(x)*x**k,a,b)[0]

def nodesAndWeights(n):
    B = np.array([[None]*(n+1)]*(n+1))
    b = np.array([None]*(n+1))
    for k in range(n+1):
        b[k] = -h(n+1+k)
        for i in range(n+1):
            B[k,i] = h(k+i)
    cs = np.linalg.solve(B,b)
    cs = np.append(cs,[1])
    xs = np.roots(cs[::-1]).real
    xs.sort()
    A = np.array([None]*(n+1))
    for k in range(n+1):
        b[k] = h(k)
        for i in range(n+1):
            B[k,i] = xs[i]**k
    As = np.linalg.solve(B,b)
    return xs, As

def gauss(f,A,x):
    return sum(A*np.array(list(map(f,x))))

xs,As = nodesAndWeights(6)
gauss(lambda x: math.exp(-x),As,xs)

# monte-carlo

n=0
N=10000

for i in range(N):
    x = random.uniform(−1,1)
    y = random.uniform(−1,1)
    if (x**2+ y**2<1):
        n = n+1

p1 = n/N
S1 = p1*2*2
error=2*2*sqrt(p1*(1−p1))/(sqrt(N))

# derivative of Lagrangian

def dpi(x,xs):
    res = 0
    for i in xs:
        res += 1/(x-i)
    return res

def pi(x,xs,i):
    xi = xs[i]
    xs.remove(xi)
    res = 1
    for j in xs:
        res *= (x-j)/(xi-j)
    return res*dpi(x,xs)

def derLag(x,xs,ys):
    res = 0
    for i in range(len(ys)):
        res += ys[i]*pi(x,xs[:],i)
    return res

xs = [1.0,1.1,1.2,1.4,1.5]
ys = [sin(x) for x in xs]

c = derLag(1.3, xs, ys)

print(c,cos(1.3),c-cos(1.3))

print(len(ns),len(y))

res1 = list(secant(f,mpf(5),mpf(3)))

n = 0
y = []
ns = []
for x in res1:
    n += 1
    ns.append(n)
    y.append((abs(log(x-res1[-1]))/log(10))/(0.5*(1+sqrt(5)))**n)

# determinant matrix

def minor(A,j):
    del A[0]
    A = list(transpose(A))
    del A[j]
    return list(transpose(A))

def det(A):
    if len(A) == 1:
        return A[0][0]
    res = 0
    for j in range(len(A[0])):
        res += (-1)**j*A[0][j]*det(minor(A[:],j))
    return res

A = [[1,19,3,4],[17,5,6,16],[7,71,9,9],[4,3,88,12]]
print(det(A))

# fibonacci

def memfib(n):
    def inner(target,total,prev,current):
        if target == total: return current
        return inner(target,total+1,current,prev+current)
    return inner(n-1,0,0,1)

# multiple precision

from mpmath import *
mp.dps  = 500
newton(f,df,mpf(1))

random.uniform(-1,1)

from numpy import *

# Euler ################

def f(x,ys):
    return -ys[0]

def foo(x,ys):
    n = ys.size
    res = zeros(n)
    for i in range(n-1):
        res[i] = ys[i+1]
    res[n-1] = f(x,ys)
    return res

def Eulr(initial,a,b,itt):
    h = (b-a)/itt
    n = initial.size
    ys = zeros((itt+1,initial.size))
    ys[0] = initial
    for i in range(itt):
        ys[i+1] = ys[i] + h*foo(a+i*h,ys[i])
    return ys

def rungKutta(initial,a,b,itt):
    h = (b-a)/itt
    ys = zeros((itt+1,initial.size))
    ys[0] = initial
    for i in range(itt):
        ys[i+1]=ys[i]+h/2*(foo(a+i*h,ys[i])+foo(a+i*h+h,ys[i]+h*foo(a+i*h,ys[i])))
    return ys

def rungKutta4(initial,a,b,itt):
    h = (b-a)/itt
    n = initial.size
    ys = zeros((itt+1,initial.size))
    ys[0] = initial
    for i in range(itt):
        o = a+i*h
        n1 = h*foo(o,ys[i])
        n2 = h*foo(o+h/2,ys[i]+n1/2)
        n3 = h*foo(o+h/2,ys[i]+n2/2)
        n4 = h*foo(o+h,ys[i]+n3)
        ys[i+1] = ys[i]+(n1+2*n2+2*n3+n4)*1/6
    return ys

start = 0
finish = 30
itterations = 40
step = (start+finish)/itterations
xs = arange(start,finish+step/2,step)
ys = rungKutta4(array([0,1]),start,finish,itterations)
plot(xs,ys)
plot(xs,[sin(x) for x in xs])

# Shooting ########

def f(x,yvec):
    return -yvec[0]

def G(u):
    a=array([0,u])
    ys = rungKutta4(a,0,30,40)
    return ys[-1][0]-1

def secant(f,x1,x2):
    a = [x1,x2]
    for i in range(3):
        a.append((a[-1]*f(a[-2])-a[-2]*f(a[-1]))/(f(a[-2])-f(a[-1])))
    return a

u0 = secant(G,0,1)[-1]

start = 0
finish = 30
itterations = 40
step = (start+finish)/itterations
xs = arange(start,finish+step/2,step)
ys = rungKutta4(array([0,u0]),start,finish,itterations)
plot(xs,ys)

############ brownian

import numpy as np

def brown(T,s):
    zs = np.arange(0,T,0.001)
    xs=np.zeros(zs.size)
    xs[0]=0
    sq = np.sqrt(0.001)
    for i in np.arange(1,zs.size):
        xs[i] = xs[i-1] + s*np.random.normal(0,sq)
    return zs, xs

import matplotlib.pyplot as plt
zs, xs = brown(1,1)
# plt.plot(zs, xs)
# plt.show()

ys = np.zeros(100)
T=1

for i in range (100):
    zs, xs = brown(T,1)
    ys[i]=xs[-1]
    # plt.plot(zs, xs,"b")
    # plt.show()

from scipy.stats.kde import gaussian_kde
kde = gaussian_kde(ys)
xs = np.arange(-5,5,0.1)
plt.plot(xs,kde(xs))
plt.plot(xs,np.exp(-xs*xs/2/T)/np.sqrt(2*np.pi*T))
plt.show()
