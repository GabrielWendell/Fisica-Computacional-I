import functools

import numpy as np
from numpy import where, insert
from numpy.linalg import det

from scipy.integrate import ode
from scipy.interpolate import splrep, splev

import matplotlib
import matplotlib.pyplot as plt

# Nossa área de interesse
XMIN, XMAX = None, None
YMIN, YMAX = None, None
ZOOM = None
XOFFSET = None

#-----------------------------------------------------------------------------
# Decoradores

def arrayargs(func):
    # Assegura que todos os args sejam arrays
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Assegura que todos os args sejam arrays
        return func(*[array(a) for a in args], **kwargs)
    return wrapper



#-----------------------------------------------------------------------------
# Funções

# pylint: disable=too-many-arguments
def init(xmin, xmax, ymin, ymax, zoom=1, xoffset=0):
    # Inicializa o domínio
    # pylint: disable=global-statement
    global XMIN, XMAX, YMIN, YMAX, ZOOM, XOFFSET
    XMIN, XMAX, YMIN, YMAX, ZOOM, XOFFSET = \
      xmin, xmax, ymin, ymax, zoom, xoffset

def norm(x):
    # Retorna a magnitude do vetor x.
    return np.sqrt(np.sum(np.array(x)**2, axis=-1))

def point_line_distance(x0, x1, x2):
    # Encontra a distância mais curta entre o ponto x0 e a linha x1 a x2.
    assert x1.shape == x2.shape == (2,)
    return np.fabs(np.cross(x0-x1, x0-x2))/norm(x2-x1)

def angle(x0, x1, x2):
    # Retorna o ângulo entre três pontos
    assert x1.shape == x2.shape == (2,)
    a, b = x1 - x0, x1 - x2
    return np.arccos(np.dot(a, b)/(norm(a)*norm(b)))

def is_left(x0, x1, x2):
    # Retorna verdadeiro se x0 estiver à esquerda da linha entre x1 e x2
    assert x1.shape == x2.shape == (2,)
    matrix = np.array([x1-x0, x2-x0])
    if len(x0.shape) == 2:
        matrix = matrix.transpose((1, 2, 0))
    return det(matrix) > 0

def lininterp2(x1, y1, x):
    # Interpolação linear nos pontos x entre matrizes numpy (x1, y1).
    # Apenas y1 pode ser bidimensional. Os valores x1 devem ser classificados de baixo para alto. Retorna um numpy.array de valores y correspondentes a pontos x. 
    
    return splev(x, splrep(x1, y1, s=0, k=1))

def finalize_plot():
    # Finaliza o plot
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.xlim(XMIN/ZOOM+XOFFSET, XMAX/ZOOM+XOFFSET)
    plt.ylim(YMIN/ZOOM, YMAX/ZOOM)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)


#-----------------------------------------------------------------------------
# Classes

class PointCharge:
    """A point charge."""

    R = 0.01  # Raio efetivo da carga

    def __init__(self, q, x):
        # Inicializa a quantidade de carga 'q' e o vetor posição 'x'.
        self.q, self.x = q, np.array(x)

    def E(self, x):  # pylint: disable=invalid-name
        # Vetor Campo Elétrico
        if self.q == 0:
            return 0
        dx = x-self.x
        return (self.q*dx.T/np.sum(dx**2, axis=-1)**1.5).T

    def V(self, x):  # pylint: disable=invalid-name
        # Potencial
        return self.q/norm(x-self.x)

    def is_close(self, x):
        # Retorna True se x estiver muito próximo da carga, caso contrário retorna False
        return norm(x-self.x) < self.R

    def plot(self):
        # Plota a carga
        color = 'b' if self.q < 0 else 'r' if self.q > 0 else 'k'
        r = 0.1*(np.sqrt(np.fabs(self.q))/2 + 1)
        circle = plt.Circle(self.x, r, color=color, zorder=10)
        plt.gca().add_artist(circle)


class PointChargeFlatland(PointCharge):
    # Uma carga pontual em Flatland.

    def E(self, x):  # pylint: disable=invalid-name
        # Vetor Campo Elétrico
        dx = x-self.x
        return (self.q*dx.T/np.sum(dx**2, axis=-1)).T

    def V(self, x):
        raise RuntimeError('Not implemented')


class LineCharge:
    # Uma linha de carga

    R = 0.01  # Raio efetivo da carga

    def __init__(self, q, x1, x2):
        # Inicializa a quantidade de carga 'q' e vetores de ponto final 'x1' e 'x2'.
        self.q, self.x1, self.x2 = q, np.array(x1), np.array(x2)

    def get_lam(self):
        # Retorna a carga total da linha
        return self.q / norm(self.x2 - self.x1)
    lam = property(get_lam)

    def E(self, x):  # pylint: disable=invalid-name
        # Vetor campo elétrico
        x = np.array(x)
        x1, x2, lam = self.x1, self.x2, self.lam

        # Obtém comprimentos e ângulos para os diferentes triângulos
        theta1, theta2 = angle(x, x1, x2), np.pi - angle(x, x2, x1)
        a = point_line_distance(x, x1, x2)
        r1, r2 = norm(x - x1), norm(x - x2)

        # Calcula as componentes paralela e perpendicular
        sign = where(is_left(x, x1, x2), 1, -1)

        # pylint: disable=invalid-name, invalid-unary-operand-type
        Epara = lam*(1/r2-1/r1)
        Eperp = -sign*lam*(np.cos(theta2)-np.cos(theta1))/where(a == 0, np.infty, a)

        # Transforma no espaço de coordenadas e retorna
        dx = x2 - x1

        if len(x.shape) == 2:
            Epara = Epara[::, np.newaxis]
            Eperp = Eperp[::, np.newaxis]

        return Eperp * (np.array([-dx[1], dx[0]])/norm(dx)) + Epara * (dx/norm(dx))

    def is_close(self, x):
        # Retorna True se x estiver próximo da carga

        theta1 = angle(x, self.x1, self.x2)
        theta2 = angle(x, self.x2, self.x1)

        if theta1 < np.radians(90) and theta2 < np.radians(90):
            return point_line_distance(x, self.x1, self.x2) < self.R
        return np.min([norm(self.x1-x), norm(self.x2-x)], axis=0) < self.R

    def V(self, x):  # pylint: disable=invalid-name
        # Potencial
        r1 = norm(x-self.x1)
        r2 = norm(x-self.x2)
        L = norm(self.x2-self.x1)  # pylint: disable=invalid-name
        return self.lam*np.log((r1+r2+L)/(r1+r2-L))

    def plot(self):
        # Plota a carga
        color = 'b' if self.q < 0 else 'r' if self.q > 0 else 'k'
        x, y = zip(self.x1, self.x2)
        width = 5*(np.sqrt(np.fabs(self.lam))/2 + 1)
        plt.plot(x, y, color, linewidth=width)


# pylint: disable=too-few-public-methods
class FieldLine:
    # Linha de campo

    def __init__(self, x):
        # Inicializa a linha de campo aponta 'x'.
        self.x = x

    def plot(self, linewidth=None, linestyle='-',
             startarrows=True, endarrows=True):
        # Plota a linha e as setas do campo.

        if linewidth is None:
            linewidth = matplotlib.rcParams['lines.linewidth']

        x, y = zip(*self.x)
        plt.plot(x, y, '-k', linewidth=linewidth, linestyle=linestyle)

        n = int(len(x)/2) if len(x) < 225 else 75
        if startarrows:
            plt.arrow(x[n], y[n], (x[n+1]-x[n])/100., (y[n+1]-y[n])/100.,
                         fc="k", ec="k",
                         head_width=0.1*linewidth, head_length=0.1*linewidth)

        if len(x) < 225 or not endarrows:
            return

        plt.arrow(x[-n], y[-n],
                     (x[-n+1]-x[-n])/100., (y[-n+1]-y[-n])/100.,
                     fc="k", ec="k",
                     head_width=0.1*linewidth, head_length=0.1*linewidth)


class ElectricField:
    # O campo elétrico devido a uma coleção de cargas

    dt0 = 0.01  # O intervalo de tempo para integrações

    def __init__(self, charges):
        # Inicializa o campo fornecido com 'cobranças'.
        self.charges = charges

    def vector(self, x):
        # Retorna o vetor de campo
        return np.sum([charge.E(x) for charge in self.charges], axis=0)

    def magnitude(self, x):
        # Retorna a magnitude do vetor de campo
        return norm(self.vector(x))

    def angle(self, x):
        # Retorna o ângulo do vetor de campo do eixo x (em radianos).
        return np.arctan2(*(self.vector(x).T[::-1])) # np.arctan2 acerta o quadrante

    def direction(self, x):
        # Retorna um vetor unitário apontando na direção do campo.
        v = self.vector(x)
        return (v.T/norm(v)).T

    def projection(self, x, a):
        # Retorna a projeção do vetor de campo em uma linha em determinado ângulo do eixo x.
        return self.magnitude(x) * np.cos(a - self.angle(x))

    def line(self, x0):
        # Retorna a linha de campo passando por x0.

        if None in [XMIN, XMAX, YMIN, YMAX]:
            raise ValueError('Domain must be set using init().')

        # Configura o integrador para a linha de campo
        streamline = lambda t, y: list(self.direction(y))
        solver = ode(streamline).set_integrator('vode')

        # Inicializa as listas de coordenadas
        x = [x0]

        # Integra nas direções para frente e para trás
        dt = 0.008

        # Resolve nas direções direta e reversa
        for sign in [1, -1]:

            # Defina as coordenadas iniciais e a hora
            solver.set_initial_value(x0, 0)

            # Integre a linha de campo em etapas sucessivas de tempo
            while solver.successful():

                # Encontra o próximo passo
                solver.integrate(solver.t + sign*dt)

                # Salva as coordenadas
                if sign > 0:
                    x.append(solver.y)
                else:
                    x.insert(0, solver.y)

                # Checa se a linha se conecta com a carga
                flag = False
                for c in self.charges:
                    if c.is_close(solver.y):
                        flag = True
                        break

                # Encerrar linha em carga ou se ela sair da área de interesse
                if flag or not (XMIN < solver.y[0] < XMAX) or \
                  not YMIN < solver.y[1] < YMAX:
                    break

        return FieldLine(x)

    def plot(self, nmin=-3.5, nmax=1.5):
        # Plot da magnitude do campo
        x, y = np.meshgrid(
            np.linspace(XMIN/ZOOM+XOFFSET, XMAX/ZOOM+XOFFSET, 200),
            np.linspace(YMIN/ZOOM, YMAX/ZOOM, 200))
        z = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                # pylint: disable=unsupported-assignment-operation
                z[i, j] = np.log10(self.magnitude([x[i, j], y[i, j]]))
        levels = np.arange(nmin, nmax+0.2, 0.2)
        cmap = plt.cm.get_cmap('plasma')
        plt.contourf(x, y, np.clip(z, nmin, nmax),
                        10, cmap=cmap, levels=levels, extend='both')


class Potential:
    # O potencial devido a uma coleção de cobranças

    def __init__(self, charges):
        # Inicializa o campo fornecido com 'cobranças'.
        self.charges = charges

    def magnitude(self, x):
        # Retorna a magnitude do potencial
        return sum(charge.V(x) for charge in self.charges)

    def plot(self, zmin=-1.5, zmax=1.5, step=0.25, linewidth=1, linestyle=':'):
        # Plot da magnitude do campo

        if linewidth is None:
            linewidth = matplotlib.rcParams['lines.linewidth']

        x, y = np.meshgrid(
            np.linspace(XMIN/ZOOM+XOFFSET, XMAX/ZOOM+XOFFSET, 200),
            np.linspace(YMIN/ZOOM, YMAX/ZOOM, 200))
        z = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                # pylint: disable=unsupported-assignment-operation
                z[i, j] = self.magnitude([x[i, j], y[i, j]])
        # levels = np.arange(nmin, nmax+0.2, 0.2)
        # cmap = plt.cm.get_cmap('plasma')
        plt.contour(x, y, z, np.arange(zmin, zmax+step, step),
                       linewidths=linewidth, linestyles=linestyle, colors='k')


# pylint: disable=too-few-public-methods
class GaussianCircle:
    # Círculo gaussiano de raio r.

    def __init__(self, x, r, a0=0):
        # Inicializa a superfície gaussiana no vetor posição 'x' e dado raio 'r'. 'a0' define um ângulo de deslocamento (em radianos) CCW 
        # do eixo x. Use isso para identificar o eixo em torno do qual o fluxo os pontos devem ser simétricos.
        self.x = x
        self.r = r
        self.a0 = a0

    def fluxpoints(self, field, n, uniform=False):
        # Retorna pontos onde as linhas de campo devem entrar / sair da superfície.
        # Os pontos de fluxo são geralmente escolhidos de forma que sejam igualmente separados no fluxo do campo elétrico. No entanto, se 'uniforme' for Verdadeiro, 
        # os pontos são equidistantes.
        # Este método requer que o fluxo esteja em x ou fora em todos os lugares no círculo (a menos que 'uniforme' seja verdadeiro).

        # Cria uma matriz densa de pontos ao redor do círculo
        a = np.radians(np.linspace(0, 360, 1001)) + self.a0
        assert len(a)%4 == 1
        x = self.r*np.array([np.cos(a), np.sin(a)]).T + self.x

        if uniform:
            flux = np.ones_like(a)

        else:
            # Obtém o fluxo através de cada ponto. Certifique-se de que os fluxos sejam : tudo dentro ou tudo fora.
            flux = field.projection(x, a)

            if np.sum(flux) < 0:
                flux *= -1
            assert np.alltrue(flux > 0)

        # Crie uma curva de fluxo integrada
        intflux = np.insert(np.cumsum((flux[:-1]+flux[1:])/2), 0, 0)
        assert np.isclose(intflux[-1], np.sum(flux[:-1]))

        # Divide a curva de fluxo integrada em n + 1 porções e calcule os ângulos correspondentes.
        v = np.linspace(0, intflux[-1], n+1)
        a = lininterp2(intflux, a, v)[:-1]

        return self.r*np.array([np.cos(a), np.sin(a)]).T + self.x