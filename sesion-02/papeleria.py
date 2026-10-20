import torch
from dimensiones import centro, norte

papeleria = torch.tensor([centro, norte])

# Esto corre al abrir papeleria.py, no cuando otra sesión lo importa.
if __name__ == "__main__":
    print(papeleria)
    print(papeleria.shape)
    print(papeleria.ndim)
    print(papeleria.numel())
    print(papeleria.dtype)
