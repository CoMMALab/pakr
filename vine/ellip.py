

import jax
import jax.numpy as jnp

# algorithms from Carlson 1994 (https://arxiv.org/pdf/math/9409227.pdf)

import jax
from jax import numpy as jnp
from jax import config
# config.update("jax_enable_x64", True)

# relative error will be "less in magnitude than r" 
r = 1.0e-15

def rf(x, y, z):

    r"""JAX implementation of Carlson's :math:`R_\mathrm{F}`

    Computed using the algorithm in Carlson, 1994: https://arxiv.org/pdf/math/9409227.pdf

     Args:
       x: arraylike, real valued.
       y: arraylike, real valued.
       z: arraylike, real valued.

     Returns:
       The value of the integral :math:`R_\mathrm{F}`

     Notes:
       ``rf`` does not support complex-valued inputs.
       ``rf`` requires `jax.config.update("jax_enable_x64", True)`
    """
    
    xyz = jnp.array([x, y, z])
    A0 = jnp.sum(xyz) / 3.0
    v = jnp.max(jnp.abs(A0 - xyz))
    Q = (3 * r) ** (-1 / 6) * v

    cond = lambda s: s['f'] * Q > jnp.abs(s['An'])

    def body(i, s):

        xyz = s['xyz']
        lam = (
            jnp.sqrt(xyz[0]*xyz[1]) 
            + jnp.sqrt(xyz[0]*xyz[2]) 
            + jnp.sqrt(xyz[1]*xyz[2])
        )

        s['An'] = 0.25 * (s['An'] + lam)
        s['xyz'] = 0.25 * (s['xyz'] + lam)
        s['f'] = s['f'] * 0.25

        return s

    s = {'f': 1, 'An':A0, 'xyz':xyz}
    # s = jax.lax.while_loop(cond, body, s)
    s = jax.lax.fori_loop(0, 10, body, s)

    x = (A0 - x) / s['An'] * s['f']
    y = (A0 - y) / s['An'] * s['f']
    z = -(x + y)
    E2 = x * y - z * z
    E3 = x * y * z

    return (
        1 
        - 0.1 * E2 
        + E3 / 14 
        + E2 * E2 / 24 
        - 3 * E2 * E3 / 44
    ) / jnp.sqrt(s['An'])


def rd(x, y, z):
    r"""JAX implementation of Carlson's :math:`R_\mathrm{D}`

    Computed using the algorithm in Carlson, 1994: https://arxiv.org/pdf/math/9409227.pdf

     Args:
       x: arraylike, real valued.
       y: arraylike, real valued.
       z: arraylike, real valued.

     Returns:
       The value of the integral :math:`R_\mathrm{D}`

     Notes:
       ``rd`` does not support complex-valued inputs.
       ``rd`` requires `jax.config.update("jax_enable_x64", True)`
    """

    xyz = jnp.array([x, y, z])
    A0 = 0.2 * (x + y + 3 * z)
    v = jnp.max(jnp.abs(A0 - xyz))
    Q = (0.25 * r) ** (-1 / 6) * v

    cond = lambda s: s['f'] * Q > jnp.abs(s['An'])

    def body(i, s):

        xyz = s['xyz']
        lam = (
            jnp.sqrt(xyz[0]*xyz[1]) 
            + jnp.sqrt(xyz[0]*xyz[2]) 
            + jnp.sqrt(xyz[1]*xyz[2])
        )

        s['An'] = 0.25 * (s['An'] + lam)
        s['t'] = s['t'] + s['f'] / (jnp.sqrt(xyz[2]) * (xyz[2] + lam))
        s['xyz'] = 0.25 * (xyz + lam)
        s['f'] = s['f'] * 0.25

        return s

    s = {'f': 1, 'An': A0, 'xyz': xyz, 't': 0}
    # s = jax.lax.while_loop(cond, body, s)
    s = jax.lax.fori_loop(0, 10, body, s)

    x = (A0 - x) * s['f'] / s['An']
    y = (A0 - y) * s['f'] / s['An']
    z = -(x + y) / 3

    E2 = x * y - 6 * z * z
    E3 = (3 * x * y - 8 * z * z) * z
    E4 = 3 * (x * y - z * z) * z * z
    E5 = x * y * z**3

    return s['f'] * (
        1 
        - 3 * E2 / 14 
        + E3 / 6 
        + 9 * E2 **2 / 88 
        - 3 * E4 / 22 
        - 9 * E2 * E3 / 52 
        + 3 * E5 / 26
    ) * s['An']**-1.5 + 3 * s['t']

def F(phi, m):
    r"""JAX implementation of the incomplete elliptic integral of the first kind 

    .. math::

        \[F\left(\phi,k\right)=\int_{0}^{\phi}\frac{\,\mathrm{d}\theta}{\sqrt{1-m{%
\sin}^{2}\theta}}]

    Without latex, it is the integral from 0 to phi of the function 1/sqrt(1-m*sin^2(theta)).

     Args:
       phi: arraylike, real valued.
       m: arraylike, real valued.

     Returns:
       The value of the complete elliptic integral of the first kind, :math:`F(\phi, m)`

     Notes:
       ``ellipfinc`` does not support complex-valued inputs.
       ``ellipfinc`` requires `jax.config.update("jax_enable_x64", True)`
    """

    c = 1.0 / jnp.sin(phi)**2
    return rf(c - 1, c - m, c)
    

def E(phi, m):
    r"""JAX implementation of the incomplete elliptic integral of the second kind 

    .. math::

        \[E\left(\phi,k\right)=\int_{0}^{\phi}\sqrt{1-m{\sin}^{2}\theta}\,\mathrm{d}%
\theta\\]

    Without latex, it is the integral from 0 to phi of the function sqrt(1-m*sin^2(theta)).

     Args:
       phi: arraylike, real valued.
       m: arraylike, real valued.

     Returns:
       The value of the complete elliptic integral of the second kind, :math:`E(\phi, k)`

     Notes:
       ``ellipeinc`` does not support complex-valued inputs.
       ``ellipeinc`` requires `jax.config.update("jax_enable_x64", True)`
    """

    c = 1.0 / jnp.sin(phi)**2
    return rf(c - 1, c - m, c) - m * rd(c - 1, c - m, c) / 3.0
