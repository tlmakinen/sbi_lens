import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax.tree_util import register_pytree_node_class
from jax_cosmo.scipy.integrate import simps

from sbi_lens.simulator.romberg import romb


@register_pytree_node_class
class RootResult:
    """Container matching the minimal TFP root-finder API."""

    def __init__(self, estimated_root):
        self.estimated_root = estimated_root

    def tree_flatten(self):
        return (self.estimated_root,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(children[0])


def find_root_chandrupatla(
    objective_fn,
    low,
    high,
    position_tolerance=1e-8,
    value_tolerance=1e-8,
    max_iterations=128,
):
    """Standalone JAX root finder with a TFP-like return shape.

    This implementation keeps a bracket [a, b] around a sign change and uses
    a secant proposal with bisection fallback. It is enough for the simulator
    use case and removes the tensorflow_probability dependency.
    """
    a = jnp.asarray(low)
    b = jnp.asarray(high)
    fa = objective_fn(a)
    fb = objective_fn(b)

    same_sign = jnp.signbit(fa) == jnp.signbit(fb)
    if bool(same_sign):
        raise ValueError("Root is not bracketed: objective_fn(low) and objective_fn(high) have the same sign.")

    root = 0.5 * (a + b)
    froot = objective_fn(root)
    converged = (jnp.abs(froot) <= value_tolerance) | (jnp.abs(b - a) <= position_tolerance)

    for _ in range(max_iterations):
        if bool(converged):
            break

        denom = fb - fa
        safe = jnp.abs(denom) > jnp.finfo(denom.dtype).eps
        secant = a - fa * (b - a) / jnp.where(safe, denom, 1.0)
        use_bisection = (~safe) | (secant <= jnp.minimum(a, b)) | (secant >= jnp.maximum(a, b))
        c = jnp.where(use_bisection, 0.5 * (a + b), secant)
        fc = objective_fn(c)

        left_has_root = jnp.signbit(fa) != jnp.signbit(fc)
        b = jnp.where(left_has_root, c, b)
        fb = jnp.where(left_has_root, fc, fb)
        a = jnp.where(left_has_root, a, c)
        fa = jnp.where(left_has_root, fa, fc)

        root = 0.5 * (a + b)
        froot = objective_fn(root)
        converged = (jnp.abs(froot) <= value_tolerance) | (jnp.abs(b - a) <= position_tolerance)

    return RootResult(estimated_root=root)


@register_pytree_node_class
class photoz_bin(jc.redshift.redshift_distribution):
    """Defines a smail distribution with these arguments
    Parameters:
    -----------
    parent_pz:

    zphot_min:

    zphot_max:

    zphot_sig: coefficient in front of (1+z)
    """

    def pz_fn(self, z):
        parent_pz, zphot_min, zphot_max, zphot_sig = self.params
        p = parent_pz(z)

        # Apply photo-z errors
        x = 1.0 / (jnp.sqrt(2.0) * zphot_sig * (1.0 + z))
        res = (
            0.5
            * p
            * (
                jax.scipy.special.erf((zphot_max - z) * x)
                - jax.scipy.special.erf((zphot_min - z) * x)
            )
        )
        return res

    @property
    def gals_per_arcmin2(self):
        parent_pz, zphot_min, zphot_max, zphot_sig = self.params
        return parent_pz._gals_per_arcmin2 * simps(
            lambda t: parent_pz(t), zphot_min, zphot_max, 256
        )

    @property
    def gals_per_steradian(self):
        """Returns the number density of galaxies in steradian"""
        return self.gals_per_arcmin2 * jc.redshift.steradian_to_arcmin2


def subdivide(pz, nbins, zphot_sigma):
    """Divide this redshift bins into sub-bins
    nbins : Number of bins to generate
    bintype : 'eq_dens' or 'eq_size'
    """
    # Compute the redshift boundaries for each bin generated
    zbounds = [0.0]
    bins = []
    n_per_bin = 1.0 / nbins
    for i in range(nbins - 1):
        zbound = find_root_chandrupatla(
            lambda z: romb(pz, 0.0, z) - (i + 1.0) * n_per_bin, zbounds[i], pz.zmax
        ).estimated_root
        zbounds.append(zbound)
        new_bin = photoz_bin(pz, zbounds[i], zbounds[i + 1], zphot_sigma)
        bins.append(new_bin)

    zbounds.append(pz.zmax)
    new_bin = photoz_bin(pz, zbounds[nbins - 1], zbounds[nbins], zphot_sigma)
    bins.append(new_bin)

    return bins
