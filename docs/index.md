HMIP solver documentation
==========================

.. math::
  \begin{eqnarray}
	  \minimize && f(\blue{x}) \\
	  \sto && g_j(\blue{x}) \leq 0, \quad j = 1, \cdots, m \\
	  &&	\blue{x_i} \in \{0, 1\}, \quad i = 1, \cdots, p < n \\
	  &&	0 \leq \blue{x_i} \leq 1, \quad i = p+1, \cdots, n
  \end{eqnarray}

where :math:`x` is the optimization variable and
:math:`P \in \mathbf{S}^{n}_{+}` a positive semidefinite matrix.

**Code available on** `GitHub <https://github.com/oxfordcontrol/osqp>`_.

.. rubric:: Citing HMIP

If you are using HMIP for your work, we encourage you to

* :ref:`Cite the related papers <citing>`
* Put a star on GitHub |github-star|


.. |github-star| image:: https://img.shields.io/github/stars/oxfordcontrol/osqp.svg?style=social&label=Star
  :target: https://github.com/oxfordcontrol/osqp


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User Documentation

   solver/index
   get_started/index
   interfaces/index
   parsers/index
   codegen/index
   examples/index
   contributing/index
   citing/index
