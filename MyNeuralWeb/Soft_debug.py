import numpy as np
import scipy
# Z = [[0.00000000e+00,  9.01270087e-10,  6.02952700e-11,  2.97182020e-10, 3.63282904e-10,  4.09698372e-11,  2.15285423e-10],
# [0.00000000e+00,  1.65029680e-10,  1.72046600e-11,  1.59640074e-11, 9.49488333e-11,  1.49339661e-11,  5.27192133e-11],
# [0.00000000e+00,  3.09155200e-10,  4.57538988e-12, -6.83904197e-11, 1.33968504e-11,  9.54590915e-12,  2.16634801e-12]]

# ZZ = [[-3.84101333e-09, -4.16582159e-09, -3.30803454e-09, -3.70221355e-09, -4.07217084e-09, -2.79393838e-09, -3.23990804e-09, -4.84860180e-09, -3.28102444e-09, -3.31175392e-09, -3.14993974e-09, -3.57976221e-09, -3.29072757e-09],
#  [-2.83762092e-09, -1.85001429e-09, -2.34955398e-09, -2.91260079e-09, -3.02820825e-09, -3.46457737e-09, -4.51222935e-09, -2.33148796e-09, -2.43440973e-09, -3.49706845e-09, -3.51153473e-09, -2.51979609e-09,-2.80227600e-09],
#  [-3.27170332e-09, -3.42380664e-09, -3.06925979e-09, -3.33235480e-09, -3.48415209e-09, -2.85113213e-09, -3.18590630e-09, -3.34472592e-09, -2.64492081e-09, -3.07804434e-09, -3.21492666e-09, -3.04354946e-09, -3.17854163e-09]]

Z_soft = [[ 0.02152854,  0.,          0.00438751,  0.0297182,   0.03632829,  0.00156501,
   0.00409698],
 [ 0.00527192,  0.,          0.0015983,   0.0015964,   0.00949488,  0.00027235,
   0.0014934 ],
 [ 0.00021663,  0.,          0.00095584, -0.00683904,  0.00133969, -0.00016465,
   0.00095459]]

B = [0.25, 1.23, -0.8]
def softmax(z):
  '''Return the softmax output of a vector.'''
  exp_z = np.exp(z)
  sum = exp_z.sum()
  softmax_z = (exp_z/sum)
  A = softmax_z
  cache = z
  return A

def softmax2(L):
    expL = np.exp(L)
    sumExpL = sum(expL)
    result = []
    for i in expL:
        result.append(i*1.0/sumExpL)
    return result

# print("S2", softmax2(A))
# print("S1",softmax(Z))

print("S1",softmax(Z_soft))
