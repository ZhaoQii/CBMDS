# example to run the function
import numpy as np

N, J = 300, 20

# the reparameterization of product attribute data is not supported. So, S >= T:
S, T = 5, 4

# simulate data to use
Delta_sim = np.random.normal(size = [300, 20])

# Delta_hat returns final estimates for ratings
P, X, Y, b, Delta_hat = CBMDS_JMR2008(Delta, S, T)

# calculate the variance-accounted-for statistics 
VAF = 1 - np.sum((Delta_hat - Delta_sim) ** 2) / np.sum((Delta_sim - Delta_sim.mean()) ** 2)
print('The VAF given Final estimates is:', VAF)
