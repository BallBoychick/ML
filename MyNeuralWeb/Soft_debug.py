import numpy as np
import scipy
# Z = [[0.00000000e+00,  9.01270087e-10,  6.02952700e-11,  2.97182020e-10, 3.63282904e-10,  4.09698372e-11,  2.15285423e-10],
# [0.00000000e+00,  1.65029680e-10,  1.72046600e-11,  1.59640074e-11, 9.49488333e-11,  1.49339661e-11,  5.27192133e-11],
# [0.00000000e+00,  3.09155200e-10,  4.57538988e-12, -6.83904197e-11, 1.33968504e-11,  9.54590915e-12,  2.16634801e-12]]

# ZZ = [[-3.84101333e-09, -4.16582159e-09, -3.30803454e-09, -3.70221355e-09, -4.07217084e-09, -2.79393838e-09, -3.23990804e-09, -4.84860180e-09, -3.28102444e-09, -3.31175392e-09, -3.14993974e-09, -3.57976221e-09, -3.29072757e-09],
#  [-2.83762092e-09, -1.85001429e-09, -2.34955398e-09, -2.91260079e-09, -3.02820825e-09, -3.46457737e-09, -4.51222935e-09, -2.33148796e-09, -2.43440973e-09, -3.49706845e-09, -3.51153473e-09, -2.51979609e-09,-2.80227600e-09],
#  [-3.27170332e-09, -3.42380664e-09, -3.06925979e-09, -3.33235480e-09, -3.48415209e-09, -2.85113213e-09, -3.18590630e-09, -3.34472592e-09, -2.64492081e-09, -3.07804434e-09, -3.21492666e-09, -3.04354946e-09, -3.17854163e-09]]

# Z_soft = [[ 0.02152854,  0.,          0.00438751,  0.0297182,   0.03632829,  0.00156501,
#    0.00409698],
#  [ 0.00527192,  0.,          0.0015983,   0.0015964,   0.00949488,  0.00027235,
#    0.0014934 ],
#  [ 0.00021663,  0.,          0.00095584, -0.00683904,  0.00133969, -0.00016465,
#    0.00095459]]

# Z3 = [[ 0.06801648,  0.04796989,  0.01059621,  0.0796161,   0.00704095,  0.21923896,
#    0.0023386,  -0.02758391, -0.04317978,  0.14922957]]


#Work
# A = [[0.04727903, 0.05165717, 0.04749058, 0.0472051,  0.048629,   0.04741267, 0.0472051 ],
#  [0.04721796, 0.04799059, 0.04728638, 0.0472051,  0.04728052, 0.04728061, 0.0472051 ],
#  [0.04719733, 0.04868726, 0.0472267,  0.0472051,  0.04688336, 0.04725024, 0.0472051 ]]

# A2 = [0.04727903, 0.05165717, 0.04749058, 0.0472051,  0.048629,   0.04741267, 0.0472051,
#  0.04721796, 0.04799059, 0.04728638, 0.0472051,  0.04728052, 0.04728061, 0.0472051,
#  0.04719733, 0.04868726, 0.0472267,  0.0472051,  0.04688336, 0.04725024, 0.0472051 ]
# B = [[0.25, 0.15, 0.23],
#      [0.25, 0.67, 0.02],
#      [0.41, 0.22, 0.73]]

# current = [[0.128434,   0.15939891, 0.12135223, 0.15714838, 0.1277097,  0.14918196,
#   0.15677481],
#  [0.13157441, 0.15668593, 0.12568084, 0.15486559, 0.1304949,  0.14833482,
#   0.15236351],
#  [0.14632592, 0.13902159, 0.14830776, 0.13948669, 0.14643272, 0.14119381,
#   0.13923151]]

Z_current = [[-0.01421951, -0.38424707,  0.,         -0.06624317, -0.2727126,  -0.01659951,
  -0.21599479],
 [-0.01168576, -0.31734796,  0.,         -0.05477136, -0.22049771, -0.02797414,
  -0.17467084],
 [ 0.00333997,  0.08999278,  0.,          0.01550427,  0.06466031,  0.00150889,
   0.05120724]]

# print(len(Z_current[1]))

# r1 = np.reshape(Z_current, (len(Z_current[1]), len(Z_current)))
# print("R@",r1)

def softmax(z):
  '''Return the softmax output of a vector.'''
  r1 = np.reshape(z, (len(z[1]), len(z)))
  A = []
  for i in r1:
    exp_z = np.exp(i)
    sum = exp_z.sum()
    softmax_z = (exp_z/sum)
    A.append(softmax_z)
    cache = z
  return np.asarray(A)

print("S1",softmax(Z_current))

#--------------Debug_pred--------------#
# P = softmax(r1)
# # print("P", P)

# # print(P[0][0])

# r2 = np.reshape(P, (7, 3))
# print("R2", r2)
# print("--------------------------------", r2[1])

# n = 0
# while n != len(current):
#   for i in r2:
#     print(i[n])
#   print("ITER")
#   n+=1

# Final = []
# for i in P:
#   Final.append(np.argmax(i))

# print("FINAL\n", Final)

#-----------------Debug-dAL------------####
# print(len(Final))
# Y = [2, 1, 0, 2, 0, 2, 0]
# arr = np.array(Y * 3).reshape(3, 7)
# # Y = np.reshape(Y * 3 , (3, len(Y)))
# print(arr)

#-----------------Debug-Softmax_back---------------------#



# dAL = [[ -6.38693026, -17.23397614, -13.61833247, -25.59949897,   1.19510167,
#   -13.64739958, -14.27043906],
#  [ -6.48698741, -16.71713008, -13.82363931, -22.92768378,   1.19067969,
#   -14.02753938, -14.3706025, ],
#  [ -7.30180235, -14.98481552, -15.84684867, -12.7893122,    1.15700586,
#   -15.87343678, -15.67148406]]

# def backward(output_gradient):
#         # This version is faster than the one presented in the video
#         n = np.size(current)
#         return np.dot((np.identity(n) - current.T) * current, output_gradient)


# print(backward(dAL))

#-------------
def softmax_batch(t):
    out = np.exp(t)
    return out / np.sum(out, axis=1, keepdims=True)

# print("SF\n", softmax_batch(Z_current))
Y = [2, 2, 0, 2, 0, 2, 1]
def to_full(y, num_classes):
    y_full = np.zeros((1, num_classes))
    y_full[0, y] = 1
    return y_full

def to_full_batch(y, num_classes):
    y_full = np.zeros((len(y), num_classes))
    for j, yj in enumerate(y):
        y_full[j, yj] = 1
    return y_full
print(Y)
print(to_full_batch(Y, 3))

print("Z - y_full\n", softmax(Z_current) - to_full_batch(Y, 3))
print(len(softmax(Z_current)[1]))