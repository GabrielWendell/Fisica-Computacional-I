import matplotlib.pyplot as plt 
import vpython as vp 
import numpy as np


# Constants
g = 9.81
l = 0.1
theta_0 = 179*np.pi/180
omega_0 = 0.0
t_0 = 0.0
t_max = 10.0
N = 5000
h = (t_max - t_0)/N

def f(r, t):
    theta = r[0]
    omega = r[1]
    ftheta = omega
    fomega = -(g/l)*np.sin(theta)

    return np.array([ftheta, fomega], float)


# Using fourth-order Runge-Kutta
tpoints = np.arange(t_0, t_max, h)
thetapoints = []
omegapoints = []
r = np.array([theta_0, omega_0], float)

for t in tpoints:
    thetapoints.append(r[0])
    omegapoints.append(r[1])
    k1 = h * f(r, t)
    k2 = h * f(r + 0.5*k1, t + 0.5 * h)
    k3 = h * f(r + 0.5*k2, t + 0.5 * h)
    k4 = h * f(r + k3, t + h)
    r += (k1 + 2*k2 + 2*k3 + k4)/6


# Plot theta vs t
def opt_plot():
    plt.grid(True, linestyle=":", color='0.50')
    plt.minorticks_on()
    plt.tick_params(axis='both', which='minor', direction='in', top=True, right=True, length=5, width=1, labelsize=15)
    plt.tick_params(axis='both', which='major', direction='in', top=True, right=True, length=8, width=1, labelsize=15)

# plt.plot(tpoints, (np.array(thetapoints, float)*180/np.pi), color='blue')
# plt.title(r'$\theta$ vs $t$', family='serif', fontsize=15)
# plt.xlabel('t (s)', family='serif', fontsize=15)
# plt.ylabel(r'$\theta$ (degrees)', family='serif', fontsize=15)
# opt_plot()
# plt.show()


# Making the animation
rod = vp.cylinder(pos=vp.vector(0, 0, 0), axis=vp.vector(l*np.cos(theta_0-np.pi/2), l*np.sin(theta_0-np.pi/2), 0), radius=l/40)
bob = vp.sphere(pos=vp.vector(l*np.cos(theta_0-np.pi/2), l*np.sin(theta_0-np.pi/2), 0), radius=l/10)

for theta in thetapoints:
    vp.rate(N // 10)
    rod.axis = vp.vector(l*np.cos(theta-np.pi/2), l*np.sin(theta-np.pi/2), 0)
    bob.pos = vp.vector(l*np.cos(theta-np.pi/2), l*np.sin(theta-np.pi/2), 0)