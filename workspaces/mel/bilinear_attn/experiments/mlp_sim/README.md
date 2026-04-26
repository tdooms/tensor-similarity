Look, We Have Exact Output Similarity under Gaussian Inputs
For an (L) layer Bilinear MLP, the expected output similarity can be expressed closed form using tensor networks. Assuming the inputs are distributed as (x\sim\mathcal N(0,I_d)), the expectation becomes just a weighted series of partial trace contractions between the models' tensors.
[
\boxed{
\mathbb E[A(x)B(x)]
\sum_{r=0}^{\lfloor n/2\rfloor}
\binom{n}{2r}^2
(2r-1)!!^2
(n-2r)!
,
\langle \tau^r A,\tau^r B\rangle.
}
]
Similarly, this extends to a residual bilinear MLP by accounting for the additional degree-indexed tensors introduced by the residual stream.
[
\boxed{
\mathbb E[A(x)B(x)]
\sum_{\substack{m_1,m_2\ m_1+m_2\ \mathrm{even}}}
\sum_{\substack{r,r'\ m_1-2r=m_2-2r'=k\ge 0}}
\binom{m_1}{2r}(2r-1)!!
\binom{m_2}{2r'}(2r'-1)!!
k!
\left\langle
\tau^r A^{[m_1]},
\tau^{r'} B^{[m_2]}
\right\rangle.
}
]
Similarly, this extends to an attention only transformer by accounting for (...uhhh)

The Expectation of Bilinear MLP outputs
For an (L)-layer bilinear MLP, each output coordinate is a homogeneous polynomial of degree
[
n=2^L.
]
We represent two such output coordinates by symmetric tensors
[
A,B\in \mathrm{Sym}^n(\mathbb R^d),
]
so that
[
A(x)=\langle A,x^{\otimes n}\rangle
A_{i_1\cdots i_n}x_{i_1}\cdots x_{i_n},
\qquad
B(x)=\langle B,x^{\otimes n}\rangle
B_{j_1\cdots j_n}x_{j_1}\cdots x_{j_n}.
]
We want the Gaussian-induced similarity
[
S(A,B)
\mathbb E_{x\sim \mathcal N(0,I_d)}[A(x)B(x)].
]
Expanding,
[
S(A,B)
A_{i_1\cdots i_n}
B_{j_1\cdots j_n}
\mathbb E[
x_{i_1}\cdots x_{i_n}
x_{j_1}\cdots x_{j_n}
].
]
By Isserlis' theorem, the (2n)-th Gaussian moment is a sum over all pairings of the indices
[
i_1,\dots,i_n,j_1,\dots,j_n.
]
Each pairing can be classified by the number (r) of internal pairs among the (A)-indices. Because the total number of (A)-indices and (B)-indices is the same, a pairing with (r) internal (A)-pairs must also have (r) internal (B)-pairs. The remaining
[
n-2r
]
pairs are cross-pairs between (A) and (B).
Define the partial trace operator
[
(\tau T)_{i_3\cdots i_n}
\sum_{a=1}^d T_{aa i_3\cdots i_n}.
]
Then (\tau^r T) is the tensor obtained after (r) internal contractions. Therefore, all pairings of type (r) contribute the same traced-tensor contraction
[
\langle \tau^r A,\tau^r B\rangle.
]
Thus
[
S(A,B)
\sum_{r=0}^{\lfloor n/2\rfloor}
c_{n,r}
\langle \tau^r A,\tau^r B\rangle.
]
We now count (c_{n,r}). On the (A)-side, choose the (2r) indices that will be internally paired and pair them:
[
\binom{n}{2r}(2r-1)!!.
]
The same count applies independently on the (B)-side. After these internal pairings, each side has (n-2r) remaining indices, which can be cross-matched in
[
(n-2r)!
]
ways. Hence
[
c_{n,r}
\binom{n}{2r}^2
(2r-1)!!^2
(n-2r)!.
]
Summing over all possible contraction levels gives
[
\boxed{
S(A,B)
\mathbb E[A(x)B(x)]
\sum_{r=0}^{\lfloor n/2\rfloor}
\binom{n}{2r}^2
(2r-1)!!^2
(n-2r)!
,
\langle \tau^r A,\tau^r B\rangle.
}
]
---
The Expectation of Bilinear MLP Outputs but when it has a residual stream
For a residual bilinear MLP,
[
h_{\ell+1}=h_\ell+B_\ell(h_\ell,h_\ell).
]
Since (h_\ell) is added back at every layer, lower-degree terms remain in the model instead of being replaced by the new bilinear term. Therefore the final output contains terms of multiple degrees:
[
y(x)=\sum_{m=1}^{N}y^{[m]}(x),
\qquad
N=2^L.
]
Let (M_A) and (M_B) be two residual bilinear MLPs. We write their outputs as
[
y_A(x)=\sum_{m=1}^{N} y_A^{[m]}(x),
\qquad
y_B(x)=\sum_{m=1}^{N} y_B^{[m]}(x).
]
Let (A^{[m]}) and (B^{[m]}) denote the coefficient tensors defined by
[
y_A^{[m]}(x)=A^{[m]}\cdot x^{\otimes m},
\qquad
y_B^{[m]}(x)=B^{[m]}\cdot x^{\otimes m}.
]

We want the Gaussian-induced output similarity
[
S(A,B)
\mathbb E_{x\sim \mathcal N(0,I_d)}
\left[
y_A(x)^\top y_B(x)
\right].
]
Expanding the residual decomposition,
[
S(A,B)
\sum_{m_1=1}^{N}
\sum_{m_2=1}^{N}
\mathbb E
\left[
\left(y_A^{[m_1]}(x)\right)^\top
y_B^{[m_2]}(x)
\right].
]
For a fixed pair ((m_1,m_2)),  the corresponding Gaussian moment has total order (m_1+m_2.)
[
\mathbb E
\left[
\left(y_A^{[m_1]}(x)\right)^\top
y_B^{[m_2]}(x)
\right]
\sum_o
A^{[m_1]}{o,i_1\dots i{m_1}}
B^{[m_2]}{o,j_1\dots j{m_2}}
,
\mathbb E
\left[
x_{i_1}\cdots x_{i_{m_1}}
x_{j_1}\cdots x_{j_{m_2}}
\right].
]
If (m_1+m_2) is odd, the term vanishes because the Gaussian is centered. Thus only degree pairs with even total degree remain.
Now fix a surviving pair ((m_1,m_2)). Isserlis' theorem expands the expectation as a sum over pairings. Unlike the homogeneous case, the two tensors may start with different input orders, so the number of internal traces on the two sides need not be the same. Suppose we trace (r) internal pairs in (A^{[m_1]}) and (r') internal pairs in (B^{[m_2]}).
Then (\tau^r A^{[m_1]}) has input order (m_1-2r), while (\tau^{r'}B^{[m_2]}) has input order (m_2-2r').
For the remaining legs to be contracted across the two models, these orders must match:
[
m_1-2r=m_2-2r'=k.
]
After these internal traces, both sides are same-order tensors, so the remaining contraction is just the same traced-tensor contraction used in the previous homogeneous case:
[
\left\langle
\tau^r A^{[m_1]},
\tau^{r'} B^{[m_2]}
\right\rangle.
]
Therefore the residual similarity has the schematic form
[
S(A,B)
\sum_{\substack{m_1,m_2\ m_1+m_2\ \mathrm{even}}}
\sum_{\substack{r,r'\ m_1-2r=m_2-2r'=k\ge 0}}
C_{m_1,m_2,r,r'}
\left\langle
\tau^r A^{[m_1]},
\tau^{r'} B^{[m_2]}
\right\rangle.
]
It remains to count the coefficient (C_{m_1,m_2,r,r'}). On the (A)-side, choose the (2r) indices that will be internally paired, and pair them:
[
\binom{m_1}{2r}(2r-1)!!.
]
On the (B)-side, the analogous count is
[
\binom{m_2}{2r'}(2r'-1)!!.
]
After these internal pairings, both sides have (k) remaining input legs. These are cross-matched in (k!) ways. Hence
[
C_{m_1,m_2,r,r'}
\binom{m_1}{2r}(2r-1)!!
\binom{m_2}{2r'}(2r'-1)!!
k!.
]
Substituting the coefficient gives
[
\boxed{
S(A,B)
\sum_{\substack{m_1,m_2\ m_1+m_2\ \mathrm{even}}}
\sum_{\substack{r,r'\ m_1-2r=m_2-2r'=k\ge 0}}
\binom{m_1}{2r}(2r-1)!!
\binom{m_2}{2r'}(2r'-1)!!
k!
\left\langle
\tau^r A^{[m_1]},
\tau^{r'} B^{[m_2]}
\right\rangle.
}
]