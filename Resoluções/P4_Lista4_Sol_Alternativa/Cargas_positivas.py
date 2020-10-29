import matplotlib.pyplot as plt
import numpy as np

import Electrostatics
from Electrostatics import PointCharge
from Electrostatics import ElectricField, Potential, GaussianCircle
from Electrostatics import finalize_plot

# pylint: disable=invalid-name

XMIN, XMAX = -40, 40
YMIN, YMAX = -30, 30
ZOOM = 6
XOFFSET = 0

Electrostatics.init(XMIN, XMAX, YMIN, YMAX, ZOOM, XOFFSET)

# Configura as cargas, o campo e o potencial elétrico
charges = [PointCharge(1, [-2, 0]),
           PointCharge(1, [2, 0]),
           PointCharge(0, [0, 0])]
field = ElectricField(charges)
potential = Potential(charges)

# Configura as superfícies gaussianas
g = [GaussianCircle(charges[i].x, 0.1) for i in range(len(charges))]
g[0].a0 = np.radians(-180)

# Cria as linhas de campo
fieldlines = []
for g_ in g[:-1]:
    for x in g_.fluxpoints(field, 12):
        fieldlines.append(field.line(x))

# Plot
plt.figure(figsize=(7, 5))
field.plot()
potential.plot()
for fieldline in fieldlines:
    fieldline.plot()
for charge in charges:
    charge.plot()
finalize_plot()
plt.show()