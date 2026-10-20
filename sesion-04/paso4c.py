from papeleria import papeleria

paso_a = papeleria.sum(dim=1)     # quito los días: queda (2, 3)
print(paso_a.shape)
paso_b = paso_a.sum(dim=2)        # ¿quito los productos?
