from .network import DensityNetwork
from .Lineformer import Lineformer


def get_network(type):
    """Factory function to get the appropriate network architecture."""
    if type == "mlp":
        return DensityNetwork
    elif type == "Lineformer":
        return Lineformer
    else:
        raise NotImplementedError(f"Unknown network type: {type}")
