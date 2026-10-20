import torch

torch.manual_seed(0)

# El catálogo: cada producto tiene un número
catalogo = {"cuadernos": 0, "plumas": 1, "mochilas": 2, "colores": 3, "tijeras": 4}

# La tabla de embeddings: 5 productos, 3 números por producto
diccionario = torch.nn.Embedding(5, 3)
print("tabla completa:", diccionario.weight.shape)

# Una clienta compró cuadernos, mochilas y otra vez cuadernos
compra = torch.tensor([catalogo["cuadernos"], catalogo["mochilas"], catalogo["cuadernos"]])
print("entrada:", compra.shape, compra.dtype)

salida = diccionario(compra)
print("salida:", salida.shape)
print(salida)

assert salida.shape == (3, 3), "3 productos comprados × 3 números por producto"
assert torch.equal(salida[0], salida[2]), "El mismo producto da el mismo renglón"
assert torch.equal(salida[1], diccionario.weight[2]), "Buscar un producto es sacar su renglón"
print("Forma y valores cuadran")

# Un lote: 2 clientas, 4 compras cada una
historiales = torch.tensor([[0, 1, 1, 3],
                            [2, 4, 0, 0]])
print("historiales:", historiales.shape, "->", diccionario(historiales).shape)
