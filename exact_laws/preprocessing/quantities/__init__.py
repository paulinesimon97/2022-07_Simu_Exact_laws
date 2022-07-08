import importlib
import os

here = os.path.dirname(os.path.realpath(__file__))


def load_quantity(name, incompressible):
    mod = importlib.import_module(f"exact_laws.preprocessing.quantities.{name}", "*")
    return mod.load(incompressible)


def load_all():
    quantities_names = [f[:-3] for f in os.listdir(here) if f[-3:] == '.py' and f != '__init__.py']
    quantities = {}
    for quantity_name in quantities_names:
        n, q = load_quantity(quantity_name, incompressible=False)
        In, Iq = load_quantity(quantity_name, incompressible=True)
        quantities[n] = q
        quantities[In] = Iq
    return quantities


QUANTITIES = load_all()
