Here is a clean standalone version.

---

## Two-Layer Quadratic Attention: Path Decomposition

Consider a two-layer quadratic-attention-only model. Let the input residual stream be

[
x_i \in \mathbb{R}^{d_{\text{model}}},
]

where (i) indexes token position. For a quadratic attention head (h), define the QK and OV circuits as

[
M^h = W_Q^h(W_K^h)^\top,
\qquad
U^h = W_V^hW_O^h.
]

A one-layer quadratic attention head contributes

[
p_i^h
=====

\sum_j
(x_i^\top M^h x_j)^2(x_jU^h).
]

After the first layer, the residual stream is

[
z_i
===

x_i+\sum_{h\in H_1}p_i^h.
]

A second-layer head (g\in H_2) then computes

[
p_i^g
=====

\sum_j
(z_i^\top M^g z_j)^2(z_jU^g).
]

To decompose this, write the residual stream before layer 2 as a sum of components:

[
z_i=\sum_{\alpha} r_i^\alpha,
]

where

[
r_i^0=x_i
]

is the direct residual stream, and

[
r_i^h=p_i^h,\qquad h\in H_1
]

are first-layer head outputs.

Substituting this into the second-layer head gives

[
p_i^g
=====

\sum_j
\sum_{\alpha,\beta,\gamma,\delta,\eta}
\left((r_i^\alpha)^\top M^g r_j^\beta\right)
\left((r_i^\gamma)^\top M^g r_j^\delta\right)
\left(r_j^\eta U^g\right).
]

Each term is indexed by

[
(\alpha,\beta,\gamma,\delta,\eta).
]

The five slots correspond to:

[
\alpha,\gamma
=============

\text{query-side inputs},
]

[
\beta,\delta
============

\text{key-side inputs},
]

[
\eta
====

\text{value-side input}.
]

If a slot is (0), the second-layer head reads directly from the original residual stream. If a slot is a first-layer head index, the second-layer head is reading information written by that first-layer head.

This gives a circuit-style decomposition:

* Query-side composed slots ((\alpha,\gamma)) correspond to **Q-composition**.
* Key-side composed slots ((\beta,\delta)) correspond to **K-composition**.
* Value-side composed slot (\eta) corresponds to **V-composition**, or the virtual-head style term discussed in the transformer-circuits framework. 

At a coarse level, each of the five slots is either direct or composed. Therefore the second-layer quadratic attention terms fall into

[
2^5=32
]

composition families.

Together with the direct path and first-layer attention terms, the two-layer model decomposes into

[
1 + 1 + 32 = 34
]

coarse families:

[
\text{direct path}
]

[
\text{first-layer head terms}
]

[
32\text{ second-layer composition families}.
]

More explicitly, the model can be written as

[
F
=

F_{\text{direct}}
+
F_{\text{layer 1}}
+
\sum_{\rho\in{0,1}^5}
F_{\rho}^{\text{layer 2}},
]

where (\rho) records whether each of

[
(\alpha,\beta,\gamma,\delta,\eta)
]

is direct or composed.

This decomposition is useful for TN similarity. If two models are decomposed as

[
F=\sum_\rho F_\rho,
\qquad
\tilde F=\sum_\sigma \tilde F_\sigma,
]

then exact functional TN similarity expands as

[
\langle F,\tilde F\rangle
=========================

\sum_{\rho,\sigma}
\langle F_\rho,\tilde F_\sigma\rangle.
]

So the full similarity can be represented as a family-by-family contribution matrix. Each entry tells us whether one model’s direct path, first-layer behavior, Q-composition, K-composition, V-composition, or mixed composition behavior aligns with the corresponding or different family in the other model.

After identifying a large family-level block, we can drill down headwise. For example, a large Q-composition/K-composition block can be expanded into specific layer-1/layer-2 head pairs to see which heads implement the aligned behavior. This gives a practical workflow: use the 34-family decomposition as a coarse map, then inspect individual heads only where the TN similarity contribution is large.
