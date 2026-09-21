import json

class qr2xyData:
    def __init__(self, q_max, r_max, eps, beta_q, gamma_r):
        self.q_max = q_max
        self.r_max = r_max
        self.eps = eps
        self.beta_q = beta_q
        self.gamma_r = gamma_r

    def save(self, filename):
        with open(filename, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, filename):
        with open(filename, "r") as f:
            return cls(**json.load(f))