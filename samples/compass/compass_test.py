from acetn.ipeps import Ipeps
from acetn.model import Model
from acetn.model.pauli_matrix import pauli_matrices
import numpy as np
import csv

class CompassModel(Model):
    def __init__(self, config):
        super().__init__(config)

    def two_site_observables(self, bond):
        observables = {}
        X,Y,Z,I = pauli_matrices(self.dtype, self.device)
        if self.bond_direction(bond) in ["+x","-x"]:
            observables["phi"] = X*X
        elif self.bond_direction(bond) in ["+y","-y"]:
            observables["phi"] = -Z*Z
        return observables

    def two_site_hamiltonian(self, bond):
        jx = self.params.get("jx")
        jz = self.params.get("jz")
        X,Y,Z,I = pauli_matrices(self.dtype, self.device)
        if self.bond_direction(bond) in ["+x","-x"]:
            return -0.25*jx*X*X
        elif self.bond_direction(bond) in ["+y","-y"]:
            return -0.25*jz*Z*Z

# function for appending measurement results to file
def append_measurements_to_file(theta, measurements, filename):
    with open(filename, "a", newline="") as file:
        row = [theta,]+[float(val) for val in measurements.values()]
        csv.writer(file).writerow(row)

# initialize an iPEPS
config = {
  "dtype": "float64",
  "device": "cpu",
  "TN":{
    "nx": 2,
    "ny": 2,
    "dims": {"phys": 2, "bond": 2, "chi": 20}
  },
}
ipeps = Ipeps(config)

# set the model
ipeps.set_model(CompassModel, params={"jz":1.0,"jx":1.0})

# sweep parameter range from theta = 0 to theta = pi/2
s_min = 0.0
s_max = 1.0
s_step = 0.025
for si in np.arange(s_min, s_max + s_step, s_step):
    theta = si*np.pi/2
    jx = np.cos(theta)
    jz = np.sin(theta)
    ipeps.set_model_params(jx=jx, jz=jz)
    ipeps.evolve(dtau=0.01, steps=1000)
    ipeps.evolve(dtau=0.001, steps=200)
    measurements = ipeps.measure()
    append_measurements_to_file(theta, measurements, "results.dat")
