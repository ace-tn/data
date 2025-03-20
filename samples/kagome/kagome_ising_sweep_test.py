from acetn.model.pauli_matrix import pauli_matrices
from acetn.model import Model

class KagomeIsingModel(Model):
    def __init__(self, config):
        super().__init__(config)

    def one_site_observables(self, site):
        X,Y,Z,I = pauli_matrices(self.dtype, self.device)
        mx = (X*I*I + I*X*I + I*I*X)/3
        mz = (Z*I*I + I*Z*I + I*I*Z)/3
        return {"mx": mx, "mz": mz}

    def one_site_hamiltonian(self, site):
        jz = self.params.get("jz")
        hx = self.params.get("hx")
        X,Y,Z,I = pauli_matrices(self.dtype, self.device)
        return -jz*(I*Z*Z + Z*Z*I) \
               -hx*(X*I*I + I*X*I + I*I*X)

    def two_site_hamiltonian(self, bond):
        jz = self.params.get("jz")
        X,Y,Z,I = pauli_matrices(self.dtype, self.device)
        match self.bond_direction(bond):
            case "-x":
                return -jz*(Z*I*I)*(I*I*Z) -jz*(I*Z*I)*(I*I*Z)
            case "+x":
                return -jz*(I*I*Z)*(Z*I*I) -jz*(I*I*Z)*(I*Z*I)
            case "-y":
                return -jz*(I*Z*I)*(Z*I*I) -jz*(I*I*Z)*(Z*I*I)
            case "+y":
                return -jz*(Z*I*I)*(I*Z*I) -jz*(Z*I*I)*(I*I*Z)

from acetn.ipeps import Ipeps
import numpy as np
import csv

# function for appending measurement results to file
def append_measurements_to_file(hx, measurements, filename):
    with open(filename, "a", newline="") as file:
        row = [hx,] + [float(val) for val in measurements.values()]
        csv.writer(file).writerow(row)

# initialize an iPEPS
dims = {
    "phys": 8, # use 8 for kagome or 4 for honeycomb
    "bond": 3,
    "chi": 30
}
config = {"TN":{"nx":2, "ny":2, "dims":dims}}
ipeps = Ipeps(config)

# set the iPEPS model (either KagomeIsingModel or HoneycombIsingModel)
ipeps.set_model(KagomeIsingModel, params={"jz":0.25, "hx":0})

# sweep parameter range from hx=0 to hx=2
hx_min = 0.0
hx_max = 2.0
hx_step = 0.1
for hx in np.arange(hx_min, hx_max, hx_step):
    ipeps.set_model_params(hx=hx/2.)
    ipeps.evolve(dtau=0.01, steps=1000)
    ipeps.evolve(dtau=0.001, steps=500)
    measurements = ipeps.measure()
    append_measurements_to_file(hx, measurements, "results.dat")
