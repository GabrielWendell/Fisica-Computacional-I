import numpy as np 
import vpython as vp 
import matplotlib.pyplot as plt 

# ------------------------------------------------------------------------------------------------------------------------------------- 

def banded(Aa,va,up,down):

    # Copia as entradas e determina o tamanho do sistema
    A = np.copy(Aa)
    v = np.copy(va)
    N = len(v)

    # Eliminação Gaussiana
    for m in range(N):

        # Fator normalizante
        div = A[up,m]

        # Primeiro, atualizamos o vetor 
        v[m] /= div
        for k in range(1,down+1):
            if m+k<N:
                v[m+k] -= A[up+k,m]*v[m]

        # Agora normalizamos a linha do pivô de A e 
        # subtraimos das inferiores 
        for i in range(up):
            j = m + up - i
            if j<N:
                A[i,j] /= div
                for k in range(1,down+1):
                    A[i+k,j] -= A[up+k,m]*A[i,j]

    # "Backsubstitution"
    for m in range(N-2,-1,-1):
        for i in range(up):
            j = m + up - i
            if j<N:
                v[m] -= A[i,j]*v[j]

    return v

# ------------------------------------------------------------------------------------------------------------------------------------- 

# Constantes
N = 6
C = 1.0
m = 1.0
k = 6.0
omega = 2.0
alpha = 2*k-m*omega**2

# Valores iniciais do array
A = np.empty([3,N],float)
A[0,:] = -k
A[1,:] = alpha
A[2,:] = -k
A[1,0] = alpha - k
A[1,N-1] = alpha - k
v = np.zeros(N,float)
v[0] = C

# Resolvendo a equação
x = banded(A,v,1,1)
print(x)

# Plot
def opt_plot():
    plt.grid()
    plt.minorticks_on()
    plt.tick_params(axis='both',which='minor', direction = "in",
                    top = True,right = True, length=5,width=1,labelsize=15)
    plt.tick_params(axis='both',which='major', direction = "in",
                    top = True,right = True, length=8,width=1,labelsize=15)

plt.figure(figsize=(8,5))
plt.plot(x,color='xkcd:blue')
plt.plot(x,'r.')
opt_plot()

# Visualização

class mass:
    def __init__(self, position):
        self.body = vp.sphere(radius=0.1, pos=position, canvas=scene)
        self.position = position
    
    def vibrate(self,index, omega, time):
            new_position = vp.vector(x[index] *np.cos(omega * time), 0,0)
            self.body.pos = self.body.pos + new_position
            
caption = """Esta animação é uma representação de um conjunto de N massas idênticas em uma reta horizontal,
unidas por molas lineares idênticas."""
scene = vp.canvas(title="Vibração em um sistema unidimensional", caption= caption)
scene.select()


# Estágio de Renderização

initial_time = 0
x *= 10 # Multiplicamos por 10 para aumentar a escala da animação
mass_0 = mass(vp.vector(x[0]*np.cos(omega*initial_time),0,0))
mass_1 = mass(vp.vector(x[1]*np.cos(omega*initial_time),0,0))
mass_2 = mass(vp.vector(x[2]*np.cos(omega*initial_time),0,0))
mass_3 = mass(vp.vector(x[3]*np.cos(omega*initial_time),0,0))
mass_4 = mass(vp.vector(x[4]*np.cos(omega*initial_time),0,0))
mass_5 = mass(vp.vector(x[5]*np.cos(omega*initial_time),0,0))



# Estágio de Animação

for time in np.arange(0,200,.1):
    vp.rate(10)
    mass_0.vibrate(0,omega, time)
    mass_1.vibrate(1,omega, time) 
    mass_2.vibrate(2,omega, time)
    mass_3.vibrate(3,omega, time)
    mass_4.vibrate(4,omega, time) 
    mass_5.vibrate(5,omega, time)