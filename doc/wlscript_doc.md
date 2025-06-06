# `PolyLogToolkit.wl` documentation

## Notation

- `x[1,...,n]` represents $x_{1}\cdots x_{n}$


## Helper functions



## Properties of multiple polylogarithms

`d`: differential. To avoid name clash, we often use other letters for the depth in the implementations

If $n_i > 1$,
```math
\partial_i\text{Li}_{n_1,\cdots,n_d}(x_1,\cdots,x_d)=\text{Li}_{n_1,\cdots,n_i-1,\cdots,n_d}(x_1,\cdots,x_d) d\log(x_i)
```
If $n_i = 1$, we have three different cases
```math
\partial_d\text{Li}_{n_1,\cdots,1}(x_1,\cdots,x_d) = \text{Li}_{n_1,\cdots,n_{d-1}}(x_1,\cdots,x_{d-1}x_d) d\text{Li}_1(x_d)
```
```math
\partial_1\text{Li}_{1,\cdots,n_d}(x_1,\cdots,x_n) = \text{Li}_{n_2,\cdots,n_d}(x_2,\cdots,x_d) d\text{Li}_1(x_1) - \text{Li}_{n_2,\cdots,n_d}(x_1x_2,\cdots,x_d)(d\text{Li}_1(x_1) + d\log(x_1))
```
```math
\partial_i\text{Li}_{n_1,\cdots,1,\cdots,n_d}(x_1,\cdots,x_d) = \text{Li}_{n_1,\cdots,\widehat{1},\cdots,n_d}(x_1,\cdots,x_{i-1}x_i,\cdots,x_d)d\text{Li}_1(x_i) - \text{Li}_{n_1,\cdots,\widehat{1},\cdots,n_d}(x_1,\cdots,x_ix_{i+1},\cdots,x_d)(d\text{Li}_1(x_i) + d\log(x_i))
```
```math
d = \sum_{i=1}^d\partial_i
```

`GoncharovInversion`: Goncharov's inversion formula for multiple polylogarithms
```math
\begin{align*}
\text{Li}&_{n_d,\cdots,n_1}(x_d^{-1},\cdots,x_1^{-1})\\
=&-\sum_{r=0}^{d-1}(-1)^{n_{r+1}+\cdots+n_d}\text{Li}_{n_r,\cdots,n_1}(x_r^{-1},\cdots,x_1^{-1})\text{Li}_{n_{r+1},\cdots,n_d}(x_{r+1},\cdots,x_d)-\\
&\sum_{r=1}^d\sum_{m_1+\cdots+m_d=n_r}(-1)^{n_r+\cdots+n_d+m_{r+1}+\cdots+m_d}\prod_{\substack{1\leq i\leq d\\i\neq r}}\binom{n_i+m_i-1}{n_i-1}B_{m_r}\left(\frac{\log(x_1\cdots x_d)}{2\pi i}\right)\\
&\frac{(2\pi i)^{m_r}}{m_r!}\text{Li}_{n_{r-1}+m_{r-1},\cdots,n_1+m_1}(x_{r-1}^{-1},\cdots,x_1^{-1})\text{Li}_{n_{r+1}+m_{r+1},\cdots,n_d+m_d}(x_{r+1},\cdots,x_d)
\end{align*}
```



## Encode and decode

Recall there is a total ordering on the generators $\mathbb H^{\text{Symb}}$ by encoding
```math
[x_{i_1\to i_2},\cdots,x_{i_d\to i_{d+1}}]_{n_1,\cdots,n_d}\leftrightsquigarrow(0^{i_1-1},n_1,\cdots,0^{i_d-i_{d-1}-1},n_d,0^{i_{d+1}-i_d-1})
```
We regard $(0^{m_1},n_1,\cdots,0^{m_d},n_d,0^{m_{d+1}})$ as $(m_1,n_1,\cdots,m_d,n_d,m_{d+1})$ in the implementation.

`encode`: turn $\text{Li}_{n_1,\cdots,n_d}(x_{i_1\to i_2},\cdots,x_{i_d\to i_{d+1}})$ into $(i_1-1,n_1,\cdots,n_d,i_{d+1}-i_d-1)$.

`decode`: turn $(m_1,n_1,\cdots,m_d,n_d,m_{d+1})$ back into $\text{Li}_{n_1,\cdots,n_d}(x_{i_1\to i_2},\cdots,x_{i_d\to i_{d+1}})$, where $i_k=m_1+\cdots+m_k+k$.

`encodingsCompare`: $(m_1,n_1,\cdots,m_d,n_d,m_{d+1})\prec(m'_1,n'_1,\cdots,m'_{d'},n'_{d'},m'_{d'+1})$
- if: $\sum n_k < \sum n'_k$
- else if: $\sum m_k + d < \sum m'_k + d'$
- else: $m_{d-r} > m'_{d'-r}$ or $n_{d-r} > n'_{d'-r}$ and the entries to the right coincide

`encodingsPriorTo`: encodings rendered by successive partial derivatives. The $\partial_{r}$-terms of $(m_1,n_1,\cdots,m_d,n_d,m_{d+1})$ include
- if $n_r>1$ or $d=1$: $(\cdots,n_r-1,\cdots)$
- else if $r=d$: $(\cdots,n_{d-1},m_d+1+m_{d+1})$
- else: $(\cdots,m_{r}+1+m_{r+1},n_{r+1},\cdots)$ and $(\cdots,m_r,n_{r+1},m_{r+1}+1+m_{r+2},\cdots)$

`IIToLi`: the $\Phi$ map

```math
\Phi(I(a; b)) = 1,\quad \Phi(I(0; \cdots; 0)) = 0
```
```math
\Phi(I(a_{0};0^{n_0-1},\cdots,a_{m},0^{n_m-1};0))=(-1)^{n_0+\cdots+n_m-1}\Phi(I(0;0^{n_m-1},a_{m},\cdots,0^{n_0-1};a_{0}))
```
```math
\Phi(I(a_{0};0^{n_0-1},\cdots,a_{m},0^{n_m-1};a_{m+1}))=\sum_{k=0}^m\sum_{p+q=n_k>1}\Phi(I(a_{0};\cdots,a_{k},0^{p-1};0))\Phi(I(0,0^{q-1},a_{k+1},\cdots;a_{m+1}))
```
```math
\Phi(I(0;0^{n_0-1},a_{1},0^{n_1-1},\cdots,a_{m},0^{n_m-1};a_{m+1}))=\sum_{i_0+\cdots+i_m=n_0-1}(-1)^{n_0+i_0+m-1}\\
\dfrac{\log^{i_0}(a_{m+1})}{i_0!}\binom{n_1+i_1-1}{n_1-1}\cdots\binom{n_m+i_m-1}{n_m-1}\text{Li}_{n_1+i_1,\cdots,n_m+i_m}\left(\frac{a_{2}}{a_{1}},\cdots,\frac{a_{m+1}}{a_{m}}\right)
```
Note: $\log^0(a_{m+1})$ is taken to be $1$ for $i_0=0$ even if $a_{m+1}=1$



## Variation matrix over $S_d(\mathbb C)$

The *complementary entry* of $(-1)^kI(0;a_{i_1},0^{m_{i_1}-1},\cdots,a_{i_k},0^{m_{i_k}-1};1)$ with respect to $(-1)^lI(0;a_{j_1},0^{p_{j_1}-1},\cdots,a_{j_l},0^{p_{j_l}-1};1)$ is
```math
(-1)^{l-k} I^{\sigma_{i_1}\sigma_0^{m_{i_1}-1}\cdots\sigma_{i_k}\sigma_0^{m_{i_k}-1}}(0;a_{j_1},0^{p_{j_1}-1},\cdots,a_{j_l},0^{p_{j_l}-1};1)
```

We regard $\sigma_{i_1}\sigma_0^{n_1-1}\cdots\sigma_{i_k}\sigma_0^{n_k-1}$ as the list $(i_1,n_1,\cdots, i_k,n_k)$ in the implementation.

`IIcode`: turn the word $(i_1,n_1,\cdots, i_k,n_k)$ into the encoding of 
```math
(-1)^kI(0;a_{i_1},0^{n_1-1},\cdots,a_{i_k},0^{n_k-1};1)=\text{Li}_{n_1,\cdots,n_k}(x_{i_1\to i_2},\cdots,x_{i_k\to d+1})
```
which is $(i_1-1,n_1,i_2-i_1-1,\cdots,n_k,d-i_k)$.

`IIdecode`: turn $(m_1,n_1,\cdots,m_d,n_d,m_{d+1})$ back into a word $(i_1, n_1, \cdots, i_k, n_k)$, where $i_r=m_1+\cdots+m_r+r$.

$\sigma_{j_1}\sigma_0^{p_{j_1}-1}\cdots\sigma_{j_l}\sigma_0^{p_{j_l}-1}$ and $\sigma_{i_1}\sigma_0^{m_{i_1}-1}\cdots\sigma_{i_k}\sigma_0^{m_{i_k}-1}$ are respectively the row and column indices of this complementary entry.

`IIVariationMatrix`: $V^{\mathbb I}$. It uses `encodingsPriorTo`  to compute the first column of the variation matrix, and it all comes down to

`complementaryEntry[w1, w2]`: the complementary entry
```math
(-1)^{l-k} I^{\sigma_{i_1}\sigma_0^{m_1-1}\cdots\sigma_{i_k}\sigma_0^{m_k-1}}(0;a_{j_1},0^{p_1-1},\cdots,a_{j_l},0^{p_l-1};1)
```
of $w_1 = (j_1,p_1,\cdots,j_l,p_l)$ and $w_2 = (i_1,m_1,\cdots,i_k,m_k)$.

Suppose $\{i_1,\cdots,i_k\}=\{j_{q_1},\cdots,j_{q_k}\}$ is a subsequence of $\{j_1,\cdots,j_l\}$ (otherwise the complementary entry is 0), then the complementary entry is

```math
(-1)^{l-k}I(0;\cdots;a_{j_{q_1}})\left(\prod_{r=1}^{k-1} I^{\sigma_0^{m_r-1}}(a_{j_{q_r}};\cdots;a_{j_{q_{r+1}}})\right)I^{\sigma_0^{m_k-1}}(a_{j_{q_k}};\cdots;1)
```
Assume
- $a_0=0$, $a_{d+1}=1$
- $i_0=j_0=0$, $i_{k+1}=j_{l+1}=d+1$ so that $q_0=0$, $q_{k+1}=l$,
- $m_0=1$

then the complementary entry is
```math
(-1)^{l-k}\prod_{r=0}^{k} I^{\sigma_0^{m_r-1}}(a_{j_{q_r}};\cdots;a_{j_{q_{r+1}}})
```
So we implement $I^{\sigma_0^{m_r-1}}(a_{j_{q_r}};\cdots;a_{j_{q_{r+1}}})$ as `Isigma[r]`. If $m_r > 1$, it is equal to
```math
\sum_{\substack{q_r\leq s<q_{r+1}-1 \\ m_r \leq p_s}}\sum_{u+v=p_s-m_r}I(a_{j_{q_r}};\cdots,a_{j_s},0^u;0)I(0;0^v,a_{j_{s+1}},\cdots;a_{j_{q_{r+1}}})
```