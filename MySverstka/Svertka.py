#штука, которая добавляет то кол-во паддингов, чтобы размерность сохранялась
n_h_prew = 128
f = 3
stride = 1

pad = 0

for i in range(0, n_h_prew + 1):
    pad = i
    if ((n_h_prew - f + 2 * pad) / stride) + 1 == n_h_prew:
        print(pad)
    else: 
        continue