import torch

#        ene    feb    mar    abr    may    jun    jul    ago     sep    oct    nov    dic
meses = [60000, 55000, 60000, 50000, 65000, 50000, 70000, 100000, 70000, 60000, 55000, 90000]
ventas = torch.tensor(meses)
print(ventas.shape)
print(ventas.sum())

trimestres = ventas.reshape(4, 3)
print(trimestres)
print(trimestres.shape)
print(trimestres.sum())

print(trimestres.sum(dim=1))   # total de cada trimestre
print(trimestres[3, 2])        # cuarto trimestre, tercer mes: diciembre
print(ventas.shape)            # ¿se descompuso la original?

a_mano = []
for t in range(4):
    total = 0
    for m in range(3):
        total = total + meses[t * 3 + m]
    a_mano.append(total)

assert trimestres.shape == (4, 3), "Deberían ser 4 trimestres × 3 meses"
assert trimestres.sum(dim=1).tolist() == a_mano, "Los totales por trimestre no cuadran con el for"
print("Trimestres comprobados")

semestres = ventas.reshape(2, 6)
print(semestres)
print(semestres.sum(dim=1))

anio = ventas.reshape(2, 2, 3)   # semestre × trimestre × mes
print(anio)
print(anio[1, 0, 2])

print(ventas.reshape(-1, 3).shape)
print(ventas.reshape(6, -1).shape)
print(ventas.reshape(2, -1, 3).shape)
print(ventas.reshape(-1).shape)

quintos = ventas.reshape(5, 3)
