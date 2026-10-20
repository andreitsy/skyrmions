# Monte Carlo simulation of our two-dimensional chiral magnet

Prepared on 2026-10-20 from the eight-page supplement to Yu et al., *Real-space observation of a two-dimensional skyrmion crystal*, Nature **465**, 901-904 (2010), DOI [10.1038/nature09124](https://doi.org/10.1038/nature09124).

Source read: [the supplied supplementary PDF](theory_pdf/41586_2010_BFnature09124_MOESM321_ESM.pdf). The Monte Carlo method is on printed pages **4-6**, Eqs. **(4)-(6)**. All eight pages, including the experimental figures, were inspected. The source PDF SHA-256 is `ec5c184c72cfbf94d08b500ca681309ff66bf7dd9d916b9cd0f5bf777c7503a2`.

The aim is to calculate the phase diagram of the Hamiltonian used in [phase_diagram_ba.jl](phase_diagram_ba.jl), with spins free to vary over a periodic two-dimensional sample. The result will be a diagram in **anisotropy a, field b, and specified temperature**, with a separate low-temperature calculation approaching the classical ground state.

This document specifies the method and its validation. Statements attributed to the supplement below describe the published procedure. Mesh choices, anisotropy, convergence criteria, and implementation details are our proposed adaptation. The implementation is [monte_carlo_2d.jl](monte_carlo_2d.jl); Section 15 records the choices it makes where this document leaves options open, and how to run it.

## 1. What the supplement establishes

### 1.1 Reading map

| Pages | Content | Use for our simulation |
|---|---|---|
| 1-2 | Sample preparation, Lorentz TEM, transport-of-intensity reconstruction, Eqs. (1)-(3) | Explains how the experimental in-plane magnetization images were obtained. These are imaging equations, separate from spin sampling. |
| 3-4, 7-8 | Defects, stripe breakup, skyrmion domains, Supplemental Figs. 1-4 | Shows the real-space patterns that the simulation should diagnose. |
| 4 | Three-dimensional exchange-DMI-field Hamiltonian, Eq. (4), and its continuum limit, Eq. (5) | Establishes the interaction and DMI convention. |
| 5 | Reduction to a two-dimensional Hamiltonian, Eq. (6), lattice sizes, pitch, boundary conditions and Metropolis sampling | Supplies the starting simulation model. |
| 5-6 | Temperature and field sweeps, equilibration budget and phase identification | Supplies a practical sampling and diagnostic strategy. |

The experimental images show stripes giving way to skyrmion domains, then a decreasing skyrmion density at larger fields. Defects and domain boundaries remain visible. Their experimental fields and temperatures require material calibration before comparison with our dimensionless parameters; for example, the 50 mT images are not a specification of our b.

### 1.2 Published model and protocol

The supplement uses a cubic-lattice model with ferromagnetic exchange, Dzyaloshinskii-Moriya interaction, and a magnetic field. Its symbol **K denotes DMI**, whereas our anisotropy is K_z. We will write the lattice DMI as D_lat and the uniaxial anisotropy as K_lat to keep them distinct.

The authors first examine layers of thickness from one lattice plane to one spiral wavelength. For those calculations, the low-temperature patterns within each layer agree with the single-plane patterns. They then remove the z bonds and use the two-dimensional Hamiltonian in Eq. (6). For a texture uniform through t layers, the effective two-dimensional couplings acquire a factor t. This reported reduction is evidence for their model and parameter range; a new film thickness or added interaction should be checked separately. [Supplement, pp. 4-5](theory_pdf/41586_2010_BFnature09124_MOESM321_ESM.pdf).

The simulation settings explicitly reported are:

| Item | Supplement |
|---|---|
| Spins | Classical, three components, unit length |
| Geometry | Square L by L lattice with periodic boundaries |
| Update | Single-spin Metropolis |
| Field direction | z, perpendicular to the plane |
| Temperature sweep | Cool at fixed DMI/exchange and field to T/J = 0.01 |
| Field sweep | Increase field at fixed temperature from zero to a polarized state |
| Equilibration | Typically within the first 10^5 MC steps |
| Measurements | The following 10^5 MC configurations |
| Skyrmion-crystal diagnostics | Real-space triangular arrangement, sixfold Fourier peaks, and changes in skyrmion number |

The supplement does **not** specify the spin proposal distribution, complete temperature ladder, field increments, random-number generator, independent-seed count, autocorrelation analysis, or an explicit normalization of an MC step. These choices must be documented in an implementation. The quoted step counts cannot be treated as a universal equilibration guarantee. [Supplement, pp. 5-6](theory_pdf/41586_2010_BFnature09124_MOESM321_ESM.pdf).

For a diagonal spiral with lattice wavevector (k,k), the printed relation is

\[
\tan k=\frac{D_{\rm lat}}{\sqrt{2}J_{\rm lat}}.
\]

The square root matters: text extraction can lose it. The reported choices imply:

| Coordinate-axis period d = 2 pi/k | L | D_lat/J_lat = sqrt(2) tan(2 pi/d) |
|---:|---:|---:|
| 6 | 36 | 2.449489743 |
| 10 | 30 | 1.027486297 |

Here d is the period along a coordinate axis for q = (k,k). The wavelength measured along the propagation direction is d/sqrt(2). These are useful choices for reproducing the published lattice calculation. A quantitative comparison with our continuum calculation additionally needs a finer-mesh study. [Supplement, p. 5](theory_pdf/41586_2010_BFnature09124_MOESM321_ESM.pdf).

### 1.3 Changes needed for our question

We add uniaxial anisotropy, preserve our definitions of a and b, and allow arbitrary two-dimensional textures. A temperature-field diagram at one anisotropy and a field-anisotropy diagram at one temperature are different slices through the same parameter space.

The supplement has no uniaxial-anisotropy term. For an additional published comparison with that term, Komatsu et al. use the same type of square-lattice bulk DMI plus -eta sum(S_z^2). Their zero-dipolar-interaction panel is the relevant starting comparison, although its finite lattice spacing and temperature affect the boundaries. [Komatsu et al., Eq. (1) and Fig. 3(a)](https://arxiv.org/html/2011.05496v3#S2).

## 2. Define the two-dimensional physical model

Our continuum energy is

\[
E[\mathbf n]=\int d^2r\left[
\frac{J_c}{2}\sum_{\mu=x,y}(\partial_\mu\mathbf n)^2
+D_c\,\mathbf n\cdot(\nabla\times\mathbf n)
-B_c n_z+K_c(1-n_z^2)
\right],\qquad |\mathbf n|=1,
\]

with partial_z n = 0. Subscript c distinguishes continuum coefficients from lattice couplings. The spins retain all three components:

\[
\mathbf S_{i,j}=(S_x,S_y,S_z),\qquad S_x^2+S_y^2+S_z^2=1.
\]

Two-dimensional refers to spatial dependence. Restricting the spin vectors to the plane would remove the skyrmions of interest.

Choose D_c > 0 and define

\[
\ell_D=\frac{J_c}{D_c},\qquad
b=\frac{B_cJ_c}{D_c^2},\qquad
a=\frac{K_cJ_c}{D_c^2}.
\]

After r = ell_D x, the energy becomes

\[
\frac{E}{J_c}=\int d^2x\left[
\frac12(\nabla\mathbf n)^2+\mathbf n\cdot(\nabla\times\mathbf n)
-b n_z+a(1-n_z^2)
\right].
\]

This is the normalization of the existing script. Positive a is easy axis along z. The initial model includes exchange, bulk DMI, field, and this anisotropy. Dipolar interactions, disorder and quantum fluctuations would introduce additional physics and parameters. See [the current energy-comparison script](phase_diagram_ba.jl).

The usual bulk cone rotates along the field direction, q parallel to z. A field n(x,y) cannot contain that z-dependent state. The Monte Carlo configuration space should still allow tilted, noncoplanar, or canted textures that vary within the plane; classification should follow the measured configurations.

## 3. Discretize without changing a and b

### 3.1 Lattice Hamiltonian

Use Nx by Ny sites and count each bond once, in the positive x and y directions:

\[
\begin{aligned}
\mathcal H={}&-J_{\rm lat}\sum_i
\left(\mathbf S_i\cdot\mathbf S_{i+\hat x}
+\mathbf S_i\cdot\mathbf S_{i+\hat y}\right)\\
&-D_{\rm lat}\sum_i\left[
\hat x\cdot(\mathbf S_i\times\mathbf S_{i+\hat x})
+\hat y\cdot(\mathbf S_i\times\mathbf S_{i+\hat y})\right]\\
&-B_{\rm lat}\sum_i S_i^z
+K_{\rm lat}\sum_i\left[1-(S_i^z)^2\right].
\end{aligned}
\]

The constant in K_lat[1-(S_z)^2] has no effect on acceptance. It is retained to match the continuum energy convention. Replacing it by -K_lat(S_z)^2 requires tracking the changed energy offset.

These DMI vectors lie **along the bonds**. They give bulk Bloch DMI, as in the supplement and our radial Bloch-profile calculation. Changing the DMI vectors to z cross bond would define interfacial DMI and a different model.

For a smooth field on a mesh of physical spacing ell,

\[
\sum_{\mu=x,y}\hat\mu\cdot
\left(\mathbf n\times\partial_\mu\mathbf n\right)
=-\mathbf n\cdot(\nabla\times\mathbf n).
\]

Consequently the negative lattice cross-product term above approaches the positive continuum D_c n dot curl n. This identity fixes the sign convention.

### 3.2 Coupling conversion

Expanding neighboring spins and replacing the site sum by an area integral gives, in **two spatial dimensions**,

\[
J_{\rm lat}=J_c,\qquad
D_{\rm lat}=D_c\ell,\qquad
B_{\rm lat}=B_c\ell^2,\qquad
K_{\rm lat}=K_c\ell^2.
\]

Let h = ell/ell_D be the dimensionless mesh spacing. Then

\[
\boxed{
\frac{D_{\rm lat}}{J_{\rm lat}}=h,\qquad
\frac{B_{\rm lat}}{J_{\rm lat}}=b h^2,\qquad
\frac{K_{\rm lat}}{J_{\rm lat}}=a h^2.
}
\]

Equivalently,

\[
\boxed{
b=\frac{B_{\rm lat}J_{\rm lat}}{D_{\rm lat}^2},\qquad
a=\frac{K_{\rm lat}J_{\rm lat}}{D_{\rm lat}^2}.
}
\]

For example, a = 0.2, b = 0.6, h = 0.25 and J_lat = 1 require

```text
J_lat = 1
D_lat = 0.25
B_lat = 0.0375
K_lat = 0.0125
```

The script's continuum choice J_c = D_c = 1 fixes units. It does not require equal exchange and DMI **lattice** couplings on every mesh. Holding B_lat and K_lat fixed while reducing D_lat would move the simulated a,b point.

### 3.3 Temperature and the energy reported for comparison

Use

\[
\theta=\frac{k_BT}{J_{\rm lat}},
\]

and set k_B = J_lat = 1 in the implementation. The acceptance denominator is then theta. A paper's T/J = 0.01 corresponds to theta = 0.01 under this convention, but changing D_lat/J_lat changes its relation to the modulation energy scale.

For N = Nx Ny spins, the saturated +z state's energy per site is

\[
e_{\rm FM}=-2J_{\rm lat}-B_{\rm lat}.
\]

The continuum-scaled energy density relative to that state is

\[
\boxed{
\rho=\frac{\mathcal H/N+2J_{\rm lat}+B_{\rm lat}}
{J_{\rm lat}h^2}.
}
\]

For smooth low-temperature states this can be compared with the script's rho in units D_c^2/J_c. At finite temperature, mean rho is an **internal-energy** comparison; equilibrium is determined by free energy.

A classical finite-temperature continuum field needs a short-distance cutoff. Refining the mesh introduces more thermally active modes. Ground-state energies have a useful smooth-mesh limit, whereas a finite-temperature phase boundary should be reported with its lattice spacing or a specified renormalization procedure. Apparent changes under mesh refinement can contain both discretization and cutoff effects.

## 4. Choose the sample size and boundary conditions

### 4.1 Periodic geometry

Wrap both coordinates:

\[
\mathbf S_{i+N_x,j}=\mathbf S_{i,j},\qquad
\mathbf S_{i,j+N_y}=\mathbf S_{i,j}.
\]

For Julia, precompute positive and negative neighbor indices with `mod1`. The dimensionless physical lengths and area are

\[
L_x=N_xh,\quad L_y=N_yh,\quad A=N_xN_yh^2.
\]

Periodic boundaries represent a repeating bulk patch of the film. Fixing the perimeter to +z would instead introduce an edge constraint and change the equilibrium question.

### 4.2 Resolve the helix and the skyrmion wall

At a = b = 0 the continuum helix has |q| = 1 and wavelength 2 pi. For the diagonal lattice helix,

\[
e(k)=-2J_{\rm lat}\cos k-\sqrt2D_{\rm lat}\sin k,
\]

whose minimum reproduces the supplement's tan(k) relation. Its wavelength in lattice sites along the propagation direction is

\[
\lambda_{\rm sites}=\frac{2\pi}{\sqrt2\arctan(h/\sqrt2)}.
\]

The following meshes keep the physical square at Lx = Ly = 36:

| h | Nx = Ny | Lattice helix wavelength, sites | Role |
|---:|---:|---:|---|
| 0.5 | 72 | 13.0736 | Inexpensive exploratory mesh |
| 0.25 | 144 | 25.3924 | Initial continuum-oriented calculation |
| 0.125 | 288 | 50.3961 | Refinement at selected points and boundaries |

These are proposed settings. Resolution must also be checked against the narrowest skyrmion wall or core encountered, which may be shorter than the helix wavelength.

Perform two distinct checks: reduce h at fixed physical lengths to test the discretization, and increase Lx,Ly at fixed h to test the sample size. For example, at h = 0.25, squares of 108, 144 and 192 sites have physical sides 27, 36 and 48.

### 4.3 Commensurability and triangular order

Allowed periodic wavevectors are qx = 2 pi mx/Lx and qy = 2 pi my/Ly. A finite box can strain a helix or favor particular orientations. Compare several winding numbers and orientations when preparing initial states.

A triangular skyrmion lattice also needs room to fit. Check rectangular boxes as well as squares. A triangular lattice of nearest-neighbor distance d_sk can fit a rectangle with sides m d_sk and n sqrt(3) d_sk for integers m,n. Approximate compatibility with the underlying mesh and vary the box; do not prescribe the skyrmion count solely to make a visually regular pattern.

The square mesh itself introduces spatial anisotropy. Its preference for a diagonal spiral at coarse spacing should decrease toward the continuum limit. Orientation locking and square-looking textures need a resolution check before physical interpretation.

## 5. Implement one Metropolis update

At fixed a,b,theta the target distribution is

\[
P(\{\mathbf S\})=Z^{-1}\exp[-\mathcal H/(k_BT)].
\]

For a symmetric proposal, accept a change with probability min[1, exp(-Delta H/kBT)]. This is the standard [Metropolis construction](https://bayes.wustl.edu/Manual/EquationOfState.pdf).

### 5.1 Propose a unit vector

A single-spin update replaces one continuous unit vector by another. It is not limited to the Ising reversal S -> -S.

For an initial implementation, mix two symmetric proposals with fixed probabilities:

1. With probability 0.95, sample uniformly in a spherical cap of half-angle delta around the current spin.
2. With probability 0.05, sample uniformly over the entire sphere.

These probabilities are practical starting choices, separate from the supplement's unspecified proposal.

For a full-sphere proposal, draw u,v uniformly from [0,1), set z = 2u-1, phi = 2 pi v, and use

\[
\mathbf S'=\left(\sqrt{1-z^2}\cos\phi,
\sqrt{1-z^2}\sin\phi,z\right).
\]

Uniform polar angle would bias the spin measure toward the poles.

For the cap, construct orthonormal e1,e2 perpendicular to S and draw

\[
c=1-u(1-\cos\delta),\qquad \phi=2\pi v,
\]

\[
\mathbf S'=c\mathbf S+\sqrt{1-c^2}
(\cos\phi\,\mathbf e_1+\sin\phi\,\mathbf e_2).
\]

A fixed cap is symmetric under exchanging S and S'. Tune delta during equilibration to avoid almost universal rejection or negligible moves; acceptance around 0.3-0.6 is a reasonable initial target. Freeze delta and the proposal mixture before production measurements. Their best values should be judged by decorrelation, not acceptance alone.

### 5.2 Compute the exact local energy change

Let Delta S = S_i' - S_i. With all four neighbors held fixed, define the field from the terms linear in S_i:

\[
\begin{aligned}
\mathbf h_i={}&J_{\rm lat}
(\mathbf S_{i+\hat x}+\mathbf S_{i-\hat x}
+\mathbf S_{i+\hat y}+\mathbf S_{i-\hat y})\\
&+D_{\rm lat}\left[
\hat x\times(\mathbf S_{i-\hat x}-\mathbf S_{i+\hat x})
+\hat y\times(\mathbf S_{i-\hat y}-\mathbf S_{i+\hat y})\right]
+B_{\rm lat}\hat z.
\end{aligned}
\]

Then

\[
\boxed{
\Delta\mathcal H=-\Delta\mathbf S\cdot\mathbf h_i
-K_{\rm lat}\left[(S_i^{\prime z})^2-(S_i^z)^2\right].
}
\]

The opposite-neighbor difference in the DMI field is essential. The anisotropy is evaluated by its exact quadratic difference. Adding 2 K_lat S_z to h_i and then treating it as a fixed linear field would give an incorrect finite-move energy change.

This calculation costs a fixed amount per attempt, independent of sample size. A sweep therefore costs O(N). Compare Delta H against a full Hamiltonian difference before using this optimization.

### 5.3 Accept and define a sweep

If Delta H <= 0, accept. Otherwise, accept when

\[
\log u<-\frac{\Delta\mathcal H}{k_BT}.
\]

Use the log comparison to avoid unnecessary exponential underflow. At T = 0 accept energy-lowering moves for minimization; equilibrium sampling and temperature-exchange formulas below require T > 0.

In this project, define **one Monte Carlo sweep, or MCS, as N attempted single-spin updates**. Random-site selection with replacement is a simple implementation. Save measurements at fixed sweep intervals, including configurations that contain rejected moves. Sampling only after accepted moves would bias the ensemble.

The supplement quotes MC steps/configurations without fully defining their counting convention. Our metadata must state the convention above explicitly.

## 6. Equilibrate through several histories

Local spin moves can remain trapped in a stripe state, a polarized state, or a particular skyrmion-number sector. A nearly constant energy trace can coexist with a frozen topological charge.

### 6.1 Prepare several initial states

At each important a,b point, use independent histories covering:

| Initial state | Construction and purpose |
|---|---|
| Random | Independent uniform-sphere spins; anneal from a disordered temperature. Tests spontaneous ordering. |
| Helix/helicoid | Start with a commensurate spiral, then allow its profile to distort. Use several orientations and winding numbers. |
| Polarized | All spins along +z, with a small random perturbation. Tests access from large field. |
| Skyrmion lattice | Seed Bloch skyrmions with several counts and spacings. Tests competing topological sectors. |

A convenient axial helix seed is S(x,y) = (0,-sin(qx),cos(qx)) for D_lat > 0 and q > 0. Choose q to fit the periodic box.

For a single Bloch seed centered at the origin, use

\[
\mathbf n(r,\phi)=
\left(\sin\beta(r)\cos(\phi+\gamma),
\sin\beta(r)\sin(\phi+\gamma),\cos\beta(r)\right),
\]

with beta(0) = pi, beta(r) approaching zero, and gamma = pi/2 for our D_c > 0 convention. A smooth profile of compact support avoids a seam when inserting separated seeds in a periodic box. For dense lattices, construct a smooth periodic texture and normalize every spin; inspect the bonds across the boundaries. A three-wave seed is another useful starting state, but its topology must be measured after normalization.

The initial radius, skyrmion count, and helicity are preparation choices. The evolution must remain free to change them. Changing the sign of DMI changes the favored Bloch helicity; helicity and topological-charge sign are distinct concepts.

### 6.2 Temperature sweeps

At fixed a,b,h:

1. Initialize random spins at a high temperature, for example theta = 2, and check that the state is disordered.
2. Cool geometrically, initially trying theta_next = 0.9 theta.
3. Re-equilibrate at each temperature. An exploratory budget of 10^4 sweeps per intermediate temperature can establish costs and relaxation trends.
4. At the target temperature, begin with 10^5 equilibration sweeps followed by 10^5 production sweeps, then extend according to the measured slow modes.
5. Repeat with independent random seeds. Heat ordered states back up as a separate check near transitions.

The numerical schedule in steps 1-4 is our starting proposal. Only the final T/J = 0.01 and approximate 10^5-step equilibration/measurement counts come from the supplement. A slower cooling schedule or larger budget may be necessary.

For comparison with a ground-state diagram, repeat at theta = 0.01, 0.005, 0.002 and, where useful, 0.001, then minimize the final textures as described in Section 10. Confirm that conclusions stop changing with the annealing schedule and final temperature.

### 6.3 Field sweeps and hysteresis

For each chosen a and target temperature, increase b from zero until the sample is polarized. At each field, equilibrate and measure. Then decrease b from a polarized state using a separate history. Compare these sweeps with temperature-annealed runs at the same points.

The supplement uses cooling and increasing-field sweeps. Decreasing-field sweeps and several seeded sectors are additional checks for our calculation. Keep the data from each history separate until they agree statistically.

If upward and downward sweeps disagree, label the interval as history dependent. The difference diagnoses a sampling problem or a metastability interval; its edges alone do not locate an equilibrium phase transition.

## 7. Measure enough observables to identify a phase

### 7.1 Energy and magnetization

Save total energy and its exchange, DMI, field and anisotropy parts. Measure

\[
m_z=\frac1N\sum_i S_i^z,\qquad
\mathbf m_\perp=\frac1N\sum_i(S_i^x,S_i^y),
\]

as well as the magnitude of the in-plane magnetization. For equilibrium samples at positive temperature,

\[
c_V=\frac{\langle\mathcal H^2\rangle-\langle\mathcal H\rangle^2}
{N k_BT^2},\qquad
\chi_z=\frac{N}{k_BT}
(\langle m_z^2\rangle-\langle m_z\rangle^2).
\]

Here chi_z is the response to the energy coupling B_lat, not directly to b or to a field in tesla. With k_B = J_lat = 1, use theta for T. Susceptibility and heat-capacity features help locate changes, but need size and sampling checks.

### 7.2 Geometric topological charge

Use the continuum convention

\[
Q=\frac{1}{4\pi}\int d^2x\,
\mathbf n\cdot(\partial_x\mathbf n\times\partial_y\mathbf n).
\]

For the lattice, triangulate every square with the same diagonal. If its vertices in counterclockwise spatial order are A = (i,j), B = (i+1,j), C = (i+1,j+1), D = (i,j+1), use triangles (A,B,C) and (A,C,D), including those crossing the periodic boundaries.

For ordered unit spins u,v,w, the oriented spherical area is

\[
\Omega(\mathbf u,\mathbf v,\mathbf w)=2\operatorname{atan2}
\left(
\mathbf u\cdot(\mathbf v\times\mathbf w),
1+\mathbf u\cdot\mathbf v+\mathbf v\cdot\mathbf w+\mathbf w\cdot\mathbf u
\right),
\]

and

\[
Q=\frac{1}{4\pi}\sum_{\rm triangles}\Omega.
\]

This is the geometric lattice-charge construction associated with [Berg and Lüscher](https://doi.org/10.1016/0550-3213(81)90568-X). It is a proposed estimator for our implementation; the supplement does not print this formula.

Use `atan2(numerator, denominator)`, or Julia's two-argument `atan(numerator, denominator)`. A one-argument arctangent loses the branch. Handle degenerate triangles explicitly and flag large neighboring-spin angles; near lattice-scale singular configurations the geometric interpolation can become ambiguous.

With this orientation and a core pointing down in an up background, the Bloch seed in Section 6 has Q = -1. Test that sign before analyzing Monte Carlo data. The opposite charge convention is also usable if it is applied consistently to every result.

Save the charge map and

\[
\rho_Q=\frac{\langle Q\rangle}{A},\qquad
\langle|Q|\rangle.
\]

For well-resolved skyrmions of one polarity, |Q| estimates their number. Opposite charges can cancel, and thermally disordered textures need more care. The sum of absolute local charges is a useful diagnostic, but it is not a reliable particle count for arbitrary noisy configurations.

### 7.3 Structure factor

Measure the equal-time spin structure factor

\[
S(\mathbf q)=\frac1N\left\langle
\left|\sum_j\mathbf S_j e^{-i\mathbf q\cdot\mathbf x_j}\right|^2
\right\rangle,
\]

and a separate S_zz(q). Subtracting the snapshot's mean S_z before its FFT removes the uniform component when displaying modulation peaks.

Average squared Fourier amplitudes of individual configurations. Taking the Fourier transform of a time-averaged texture can erase a translating or rotating lattice.

Typical signatures are a pair of dominant nonzero wavevectors for stripes, and three pairs with roughly sixfold angular arrangement for a triangular skyrmion lattice. Distorted stripes have higher harmonics. A disordered skyrmion state can have nonzero Q and broad Fourier weight without crystal order.

The supplement uses real-space images, Fourier peaks, and skyrmion number together. Preserve that combined criterion. [Supplement, p. 6](theory_pdf/41586_2010_BFnature09124_MOESM321_ESM.pdf).

### 7.4 Skyrmion positions and orientational order

Locate core domains, using both S_z and the local charge map to distinguish cores from stripe segments or thermal fluctuations. Account for connected regions crossing periodic boundaries when computing their centers.

For neighboring skyrmion centers, a sixfold bond-order diagnostic is

\[
\psi_6=\frac1{N_{\rm sk}}\sum_j
\frac1{z_j}\sum_{k\in\mathcal N_j}e^{6i\phi_{jk}}.
\]

Use periodic neighbors, for example a periodic Delaunay construction, and measure <|psi_6|> as well as its spatial correlations. If no cores are identified, mark this observable as undefined.

This helps distinguish a triangular arrangement from a skyrmion liquid or multiple domains. Finite-temperature order in two dimensions requires size and correlation analysis before claiming thermodynamic long-range crystal order. Komatsu et al. explicitly complement chirality with orientational diagnostics. [Komatsu et al., Sec. II](https://arxiv.org/html/2011.05496v3#S2).

### 7.5 Save real-space snapshots

For representative equilibrated configurations, save the complete spin field and plot S_z, sparse in-plane arrows, and the local topological-charge density. Include a,b,theta,h,Lx,Ly,seed and history in the metadata or caption. Preserve the x,y coordinate orientation when exporting images so that visual handedness remains consistent with the numerical charge convention.

## 8. Check equilibration and statistical uncertainty

Record time series of energy, m_z, Q, and the relevant structural order parameter. Examine traces from every preparation history.

For measurements of observable O at fixed intervals, define the normalized autocorrelation C_O(t) and integrated time

\[
\tau_{{\rm int},O}=\frac12+\sum_{t=1}^{W}C_O(t),\qquad
N_{{\rm eff},O}\simeq\frac{M}{2\tau_{{\rm int},O}},
\]

where M is the number of recorded measurements and W is a controlled truncation window. Express tau in recorded-sample intervals or convert it to sweeps explicitly.

Block the data on scales exceeding the autocorrelation time and use blocking or a block bootstrap for uncertainties. Retain enough blocks to estimate an error meaningfully. Measure error bars for different observables separately, because energy may decorrelate much faster than Q.

Before assigning a phase, require the following evidence:

- The production trace has no detectable drift on the observed timescale.
- Independent seeds and relevant starting states give compatible observables and structures.
- Increasing the equilibration and measurement budgets leaves the conclusion within its uncertainty.
- Competing skyrmion counts and stripe windings have been explored near a boundary.
- Selected points pass size, aspect-ratio and mesh checks.

A constant Q trace does not establish a short autocorrelation time. If alternative seeded Q sectors persist, mark sector equilibration as unresolved even when the energy error within each sector is small. Saving less often reduces storage but does not cure a trapped chain.

## 9. Build the a,b phase diagram

Fix the temperature, mesh, and box for each diagram. Start with our existing window, 0 <= a <= 0.8 and 0 <= b <= 1, and include a larger field when needed to establish the polarized reference.

Use Delta a = Delta b = 0.05 for the first survey: this is 17 by 21 = 357 parameter pairs. First establish the protocol on a handful of points. Refine transition bands to 0.01 and then 0.005 after the sampling is reliable. The full 0.005 grid has 32,361 pairs, each now requiring many spin updates and several histories.

The initial classification should allow:

| Label | Evidence |
|---|---|
| Helical/helicoidal | Stripe structure, dominant pair of modulation wavevectors, little net topological charge. Allow field-induced distortion. |
| Triangular skyrmion lattice | Resolved skyrmion cores, nonzero charge density, approximately sixfold Fourier pattern, and orientational order. |
| Polarized | Large positive magnetization and no persistent modulation or skyrmion cores. Thermal fluctuations keep m_z below exactly one. |
| Disordered skyrmions / mixed texture | Skyrmions present without established triangular order, or persistent stripe/skyrmion domains. |
| Other ordered texture | Reproducible canted, square, or other structure requiring its own analysis. |
| Unresolved | Histories disagree or the sampling and size evidence is insufficient. |

Use a horizontal and b vertical, matching our current plots. Add the temperature and geometry to the figure title. Keep unresolved points visible and show uncertainty intervals on transition lines.

The supplement's H+Sk and FM+Sk labels describe defects and progressively diluted skyrmion textures. Reproducing such textures does not by itself prove an additional equilibrium phase with a sharp thermodynamic boundary. [Supplement, p. 6](theory_pdf/41586_2010_BFnature09124_MOESM321_ESM.pdf).

## 10. Compare with the current classical diagrams

### 10.1 Ground-state search

The existing script compares a small set of classical energy densities at T = 0. To compare with it:

1. Anneal from several textures and topological sectors to a low temperature.
2. Save multiple promising configurations from each history.
3. Minimize their full two-dimensional lattice energy with the unit-spin constraint.
4. Compare the resulting energy densities using the conversion in Section 3.
5. Repeat selected points on finer meshes and different boxes.

For a constrained gradient method,

\[
\mathbf g_i=\frac{\partial\mathcal H}{\partial\mathbf S_i}
=-\mathbf h_i-2K_{\rm lat}S_i^z\hat z,
\qquad
\mathbf g_i^\perp=\mathbf g_i-(\mathbf S_i\cdot\mathbf g_i)\mathbf S_i.
\]

Take a step along -g_perp, normalize the spins, and use an energy-decreasing line search. Record the maximum tangent-gradient norm and energy convergence. A converged local minimum is a candidate; agreement among multiple initial states and sectors strengthens the ground-state search.

Skyrmions can become elongated or merge into stripes during this relaxation. Such deformations are allowed in two dimensions and excluded by the script's radial profile ansatz. Survival after a finite run alone does not prove stability; test perturbations and, where needed, the constrained Hessian.

### 10.2 Relate the measured density to the fixed disk radius

The script imposes one skyrmion on a disk of radius R. An unconstrained periodic simulation lets the density adjust, subject to finite-size and sampling limitations.

For N_sk well-separated skyrmions in dimensionless area A, define the area-equivalent cell radius

\[
\boxed{R_{\rm eff}=\sqrt{\frac{A}{\pi N_{\rm sk}}}.}
\]

For a triangular arrangement,

\[
\frac{\sqrt3}{2}d_{\rm sk}^2=\pi R_{\rm eff}^2,
\qquad d_{\rm sk}\simeq1.904626R_{\rm eff}.
\]

R_eff is a **cell radius**, distinct from a core radius determined by S_z = 0. At physical box size 36 by 36, an R_eff near 4.5 corresponds to roughly 20 skyrmions, while R_eff near 20 corresponds to roughly one. The latter is insufficient for identifying a well-developed crystal; a density that dilute needs a larger sample.

To study a fixed density, prepare several N_sk sectors, minimize each, and compare their energy per area. A constraint that forces Q to remain fixed creates a separate constrained calculation. It should not be used silently while claiming unrestricted equilibrium.

### 10.3 Expected differences from the present script

Numerical boundaries can change because the Monte Carlo model has no z-dependent cone, allows non-axisymmetric textures and varied skyrmion densities, includes thermal fluctuations at positive temperature, and has a finite lattice cutoff and box.

At zero field in our positive-a window, test whether annealed and minimized configurations return to helicoidal textures. A skyrmion seeded at b = 0 may be long lived or locally stable even if its energy exceeds the helicoid. Determine energy ordering and local stability separately.

Before using a radial profile as an energy benchmark, check its solver residual and convergence flag. The present script can include a trial-profile fallback after Newton failure. A Monte Carlo comparison should use validated continuum profiles. Inspect `solve_profile` and `skyrmion_fixed_R` in [the script](phase_diagram_ba.jl).

## 11. Resolve difficult finite-temperature competition

### 11.1 Parallel tempering

If local Metropolis runs remain trapped, simulate replicas at several positive temperatures with the same a,b,h and box. Alternate ordinary local sweeps with exchanges between neighboring temperatures. For total energies E_m,E_n and beta_m = 1/(kBT_m), the acceptance is

\[
P_{\rm swap}=\min\left[1,
\exp\{(\beta_m-\beta_n)(E_m-E_n)\}\right].
\]

Use total energies, including anisotropy. Tune the temperature ladder from overlapping energy distributions and swap acceptance. A useful ladder permits replicas to travel repeatedly from low to high temperature and back. Verify that these trips actually help change Q and stripe winding; temperature exchanges alone do not guarantee sector equilibration.

Keep measurements assigned to the **temperature**, regardless of which configuration currently occupies it. Freeze the ladder before production. This extension follows the exchange-Monte-Carlo construction of [Hukushima and Nemoto, Eqs. (2.4)-(2.7)](https://arxiv.org/pdf/cond-mat/9512035#page=3), rather than the supplement's reported update method.

The number of required temperatures can grow substantially with system size. Choose it from overlap and mixing measurements, rather than the number of available CPU cores.

### 11.2 Free energy rather than the lowest mean energy

At positive temperature, different phases can have different entropies. Selecting the smallest sampled mean energy is not an equilibrium phase-selection rule.

If an equilibrated simulation repeatedly visits two well-defined phase basins A and B, their probabilities give

\[
F_A-F_B=-k_BT\log(P_A/P_B).
\]

If switching remains inaccessible, use an enhanced-sampling or free-energy method at selected disputed points, such as umbrella sampling followed by unbiasing. Q alone cannot distinguish two Q = 0 phases, so the chosen collective variables may also need stripe order or magnetization.

Thermodynamic integration provides another route if the competing branches can be tied to a common free-energy reference. For the Hamiltonian as written, define f = F/(N J_lat h^2). At fixed theta,h,

\[
\frac{\partial f}{\partial b}=-\langle m_z\rangle,\qquad
\frac{\partial f}{\partial a}=1-\left\langle(S_z)^2\right\rangle,
\]

where the second expectation includes a site average. These derivatives alone do not supply the unknown offset between separately trapped branches. Include that reference and uncertainty before quoting a free-energy crossing.

## 12. Implementation outline

The following is pseudocode for a new implementation. Names in it are proposed functions, not commands currently available in the repository.

```text
read configuration: a, b, theta_target, h, Nx, Ny, seed, history
set J_lat = 1
set D_lat = h
set B_lat = b*h*h
set K_lat = a*h*h
construct periodic neighbor tables
create a private random-number generator
initialize unit spins from the selected history
compute total energy from positive-direction bonds

for theta in the chosen annealing or fixed-temperature schedule:
    tune proposal angle only during an adaptation stage
    freeze proposal settings

    for sweep in equilibration:
        repeat Nx*Ny times:
            choose a random site
            propose a unit vector from the symmetric proposal
            compute exact local DeltaH from four neighbors and anisotropy
            accept if DeltaH <= 0 or log(random()) < -DeltaH/theta
            on acceptance: replace the spin and add DeltaH to cached energy
        periodically verify cached energy against a full recalculation

    if this is a temperature at which data are requested:
        start fresh production accumulators
        for sweep in production:
            perform the same Nx*Ny attempted updates
            at fixed intervals:
                measure energy components and magnetization
                measure Q and charge density
                measure structure factors and structural order
                append time-series data
            occasionally save a spin configuration and an exact restart

estimate autocorrelation times and block uncertainties
compare histories before phase classification
flag unresolved sector or structural competition
optionally minimize saved configurations for the ground-state study
write metadata, observables, snapshots, diagnostics and summary
```

For Julia, store spins in contiguous floating-point arrays, precompute neighbors, and avoid small heap allocations and FFTs inside the attempted-update loop. Keep the random-number stream private to each independent simulation. Run expensive structural measurements less frequently than local energy and magnetization measurements.

For parallel work, distribute independent parameter points, seeds, histories, or temperature replicas. Sequential field and cooling histories carry their configurations forward in the specified order. Inside one configuration, arbitrary simultaneous updates of neighboring spins using old local fields do not implement this Metropolis chain. A checkerboard scheme is possible for nearest-neighbor interactions when **both periodic lattice dimensions are even**, with synchronization between colors.

## 13. First calculation and validation

### 13.1 Establish a small pilot

Use J_lat = 1, h = 0.25, Nx = Ny = 144, and initially target theta = 0.01. Investigate these points before a full parameter grid:

| a | b | Purpose |
|---:|---:|---|
| 0 | 0 | Check helix pitch, energy, and orientation |
| 0 | 0.3 | Explore a field-induced two-dimensional skyrmion candidate without the bulk cone |
| 0.2 | 0.6 | Connect to the script's default comparison point |
| 0.4 | 0.2 | Examine stripe/skyrmion competition with appreciable anisotropy |
| 0.8 | 1.2 | Establish a high-field polarized reference |

These points select different physical regimes for inspection; they are not new predicted phase assignments. Start from the four preparation types in Section 6 and repeat disputed cases with independent seeds.

If this box is too costly during development, establish the update and observables on smaller samples first. Use them for correctness and timing, then restore a sample large enough to assess the structures.

As an initial recording schedule, save energy and magnetization every 10 sweeps, charge and structural observables every 100 sweeps, and a complete spin snapshot every 5,000 sweeps. These are storage and measurement-cost choices to adjust after timing and autocorrelation analysis. A saved snapshot is not automatically an independent sample. Record acceptance and the frozen proposal angle for every trajectory.

### 13.2 Correctness checks

Before interpreting a phase diagram, verify:

1. **Unit length:** spin norms stay within floating-point tolerance.
2. **Local energy:** optimized Delta H matches H_new - H_old for random changes, including sites adjacent to periodic boundaries.
3. **Energy counting:** uniform +z spins have e = -2 J_lat - B_lat. There are exactly 2N forward bonds.
4. **DMI handedness and pitch:** the diagonal helical energy agrees with the analytic expression in Section 4, and its optimum approaches |q| = 1 as h decreases and commensurability improves.
5. **Charge convention:** a smooth periodic compact Bloch seed has Q = -1 under our orientation; a uniform state has Q = 0.
6. **Thermal sampler:** with D_lat = K_lat = B_lat = 0, a small sample approaches the expected disordered limit at large temperature. With J_lat = D_lat = K_lat = 0, independent spins in a field give m_z = coth(x) - 1/x, x = B_lat/(kBT), providing a direct acceptance/proposal check.
7. **Repeatability:** the same seed and serial update order reproduce a run; independent seeds agree within sampling uncertainty.

For the uncoupled-spin test, keep the reference energy and temperature units fixed when turning off exchange. Do not redefine the temperature unit using the vanishing test coupling.

The local-energy formula and charge orientation in this guide were checked independently while preparing it. Across 200 random changes the largest local/full energy-difference discrepancy was 8.0e-15. The uniform-state and commensurate diagonal-helix energy checks agreed to floating-point precision, and a compact Bloch seed gave Q = -1. These checks validate the written formulas, not a future Monte Carlo implementation or its equilibration.

### 13.3 Budget from measured update rates

At 144 by 144 sites, 10^5 equilibration plus 10^5 production sweeps require approximately **4.15 billion attempted updates per trajectory**, before intermediate annealing temperatures. Four trajectories at each of 357 grid points already require approximately **5.92 trillion attempts** for those final stages.

Measure attempted updates per second on the intended hardware after compilation. Estimate costs from that rate and from the measured time needed to decorrelate Q and structural order. The 0.005 grid should follow a successful pilot and boundary refinement study.

## 14. Save outputs and monitor computational efficiency

Use a separate run directory, consistent with our existing output policy. This is a proposed layout:

```text
outputs/monte_carlo_2d/<run-id>/
    metadata.toml
    observables.csv
    summaries.csv
    checkpoints/
    snapshots/
    plots/
        phase_diagram.png
        convergence.png
        structure_factors.png
        spin_textures.png
```

Metadata should include continuum a,b; all four lattice couplings; h,Nx,Ny; temperature units and schedule; seed and generator; initial state and sweep direction; proposal type and frozen angle; sweep definition; equilibration and measurement lengths; measurement interval; precision; code revision/hash; hardware; wall time; and thread/process settings. A restart also needs the current spins, temperature/sweep position, and complete random-number-generator state.

Keep time series for each seed and history. The summary should contain means, block errors, autocorrelation/effective-sample estimates, energy components, magnetization, charge density, structural diagnostics, measured period or cell radius, and an equilibration flag. Generate the final phase label from these records rather than storing only a color or integer ID.

For a Slurm allocation, record whole-job CPU utilization as

\[
\mathrm{CPU\ utilization}=100\,
\frac{\mathrm{TotalCPU\ in\ seconds}}
{\mathrm{ElapsedRaw}\times\mathrm{AllocCPUS}}.
\]

Use one consistent accounting scope so that parent and step totals are not counted twice. Short live samples describe a particular interval; whole-job accounting also includes startup and output time.

Also report attempts per second and effective samples per wall-clock hour for the slow observable. High CPU utilization means the allocated CPUs are busy. It does not establish good Monte Carlo mixing or a converged phase diagram. Monte Carlo sweeps are sampling iterations and cannot be converted directly into physical spin-dynamics time or skyrmion lifetimes.

## 15. Implementation: monte_carlo_2d.jl

[monte_carlo_2d.jl](monte_carlo_2d.jl) is a self-contained CLI script in the `scripts/` environment (Plots only; the structure factor uses two dense DFT matrix products instead of an FFT). It computes a **(b, θ) phase diagram at fixed anisotropy a**, the slice of parameter space complementary to the (b, a) ground-state diagram of [phase_diagram_ba.jl](phase_diagram_ba.jl).

```bash
julia --project=scripts -e 'using Pkg; Pkg.instantiate()'      # once
julia --project=scripts -t auto monte_carlo_2d.jl --selftest    # the checks of Sec. 13.2 (~10 s), exit code 1 on failure
julia --project=scripts -t auto monte_carlo_2d.jl --bench       # attempted updates per second on this machine
# quick test of the whole pipeline, ~30 s on 8 threads (48², 11 × 12 grid, 300 + 300 sweeps per point)
julia --project=scripts -t auto monte_carlo_2d.jl --L 48 --nb 11 --ntheta 12 --neq 300 --nprod 300 --nquench 100
julia --project=scripts -t auto monte_carlo_2d.jl               # default survey (a = 0, h = 0.5, 72 × 72), ~5 min on 8 threads
julia --project=scripts monte_carlo_2d.jl --help                # all options
# cluster scale: the Sec. 13.1 mesh, ≈ 6 × 10¹² attempts ≈ 1.5 h on 64 cores
julia --project=scripts -t $SLURM_CPUS_PER_TASK monte_carlo_2d.jl --h 0.25 --L 144 --db 0.05 --ntheta 40 \
      --neq 20000 --nprod 20000 --final-factor 5 --nstruct 200 --nseeds 2 --no-timeseries --run-id h0p25_L144
```

Always use `-t auto` (or `-t N`): trajectories run on a thread pool. The cost is (trajectories) × (points per trajectory) × (`neq` + `nprod`) × Nx × Ny attempts at ≈ 18 M attempts/s per core; the script prints the total before starting. Memory per (history, seed, b, θ) record is two Float32 maps of Nx × Ny plus the time series, and `observables.csv` has `nprod/nmeas` rows per record, so use `--no-timeseries` for large runs. Keep tests small (the 48² command above); run the large meshes on the cluster. Each run writes `outputs/monte_carlo_2d/<run-id>/` with `metadata.toml`, `summaries.csv` (one row per history, seed, b, θ with every diagnostic below), `phase_diagram.csv`, `observables.csv` (time series), `snapshots/` (full spin fields at the lowest θ), and `plots/{phase_diagram, spin_textures, structure_factors, observables, convergence}.png`.

### 15.1 Choices left open by the supplement (Sec. 1.2)

| Item | Implementation |
|---|---|
| Lattice couplings | `J_lat = 1`, `D_lat = h`, `B_lat = b h²`, `K_lat = a h²` (Sec. 3.2); `--h`, `--L` or `--Nx/--Ny`, `--a` |
| Proposal (Sec. 5.1) | spherical cap of half-angle δ with probability 0.95, uniform sphere with 0.05 (`--pfull`); δ is adapted in batches of 10 sweeps during the first `--nadapt` sweeps at every (b, θ) point towards an acceptance of 0.3–0.6, then frozen; the frozen δ and the production acceptance are the `delta` and `acc` columns |
| ΔH | exact local form of Sec. 5.2 with the quadratic anisotropy difference; the cached energy is compared with a full recalculation before each production stage (`energy_check` column) |
| Sweep (Sec. 5.3) | one MCS = `Nx·Ny` attempted updates. Default order: checkerboard, colour 0 then colour 1, lattice order within a colour; `--site-order random` draws sites with replacement |
| Random numbers | one `Xoshiro` stream per trajectory (and per row block when the checkerboard is threaded), seeded from `hash((seed, history, seed index, first b index, first θ index[, block]))`; `--seed` sets the base |
| Temperature ladder (Sec. 6.2) | geometric from `--theta-max` 1.5 to `--theta-min` 0.02 in `--ntheta` 24 steps (ratio 0.83), or `--ladder linear`; every ladder point is a row of the diagram |
| Field increments | the b grid itself (`--bmin/--bmax/--nb` or `--db`) |
| Histories (Sec. 6.1, 6.3) | `cool`: random spins at θ_max, cooled down the ladder (the supplement's protocol). `heat-helix`, `heat-skx`, `heat-fm`: an axial helix with the lowest lattice-energy winding, a commensurate triple-q Bloch skyrmion lattice with `nx·my` cells (default from `h` and the box; `--skx-cells`), or the +z state with 5 % transverse noise, equilibrated at θ_min and heated up the ladder. `field-up` / `field-down`: b increased from a helix seed / decreased from the polarized state at fixed θ. Default `cool,heat-helix,heat-skx,heat-fm` × `--nseeds 1` |
| Budget | `--neq` 1000 (including adaptation) + `--nprod` 1000 sweeps per point, `--final-factor` at θ_min, then `--nquench` 500 T = 0 sweeps on a copy of the θ_min configuration (Sec. 10.1); the supplement's 10⁵ + 10⁵ sweeps are a cluster-scale setting |
| Measurements (Sec. 7) | energy components, m_z, \|m_⊥\|, ⟨S_z²⟩ every `--nmeas` 10 sweeps; Q and its map, S(q), S_zz(q), cores and ψ6 every `--nstruct` 50 sweeps |
| Structure factor | S(q)/N = \|Σ_j (S_j − S̄) e^{−iq·x_j}\|² / N² summed over components and averaged over snapshots (Parseval: Σ_q S(q)/N ≤ 1; a perfect helix gives 1/2 at ±q) |
| Cores and ψ6 (Sec. 7.4) | connected regions of S_z < `--s-core` whose touching plaquettes carry \|Q\| ≥ `--qcore-min` (a core bounded by the S_z = 0 contour carries 1/2, a stripe segment 0); neighbours within 1.35 × the median nearest-core distance instead of a periodic Delaunay construction |
| Statistics (Sec. 8) | τ_int with truncation at the first non-positive C(t) or t ≥ 6τ, blocking errors with blocks ≥ 2τ, N_eff, and a drift flag (first and second half of production within 3σ) |

### 15.2 Phase labels (Sec. 9) and how histories are combined

Per record: skyrmions are *present* when `|⟨Q⟩| ≥ --qmin` (3), `|⟨Q⟩| > 3 σ_Q`, and `std(Q) ≤ |⟨Q⟩|/2` (persistent number; the charge of a paramagnet fluctuates strongly). A **Skyrmion crystal** additionally needs `max S(q)/N ≥ --peak-min` (0.02), three dominant peak pairs (≥ `--dom-frac` 0.3 of the strongest, within [0.6, 1.4] |q₁|) about 60° apart (`--angle-tol` 15°), and `⟨|ψ6|⟩ ≥ --psi6-min` (0.5) when cores were found; otherwise **Skyrmions (disordered)**. Modulated states without persistent charge are **Helical** (single- or multi-domain). Unmodulated states are **Polarized** if `⟨m_z⟩ ≥ --mz-pol` (0.7), else **Paramagnet**; this is a crossover line, not a transition. Harmonics of distorted helicoids (2q, 3q) and the √3q shell of the crystal fall outside the dominant window.

Per grid point: the histories and seeds agree when their labels fall in one class ({Paramagnet, Polarized}, {Helical}, {Skyrmion crystal, Skyrmions (disordered)}). Disagreeing cells take the label of the record with the lowest mean energy per site and are marked with a dot in the plot and `agree = false` in `phase_diagram.csv` (`--resolve energy`, default), or are painted **Unresolved** (`--resolve strict`). The energy rule is the Sec. 10.1 ground-state comparison applied at each θ; it is a low-temperature proxy, not a free-energy criterion (Sec. 11.2). Seeded histories are metastable through most of the ordered region, so at low θ almost every cell is decided this way and is marked.

### 15.3 Parallelization (Sec. 12)

Trajectories are independent and are taken from a channel by one worker task per thread. Threads left over when there are fewer trajectories than threads (single-point runs) are used inside the lattice: the checkerboard colours alternate, and the rows of one colour are split into contiguous blocks updated by separate tasks with private random streams (`--inner-threads`; automatic only when a block holds ≳ 4096 sites, because at 72 × 72 the task overhead costs more than it saves). The random-site order cannot be threaded. BLAS runs single-threaded. Measured on an Apple M1 (4 performance + 4 efficiency cores): ≈ 18 M attempts/s per thread, ≈ 66–70 M/s aggregate with `-t 8`; the default survey (84 trajectories × 24 points × 2000 sweeps × 5184 sites ≈ 2 × 10¹⁰ attempts) takes ≈ 5 min, and ≈ 3 min for `--histories cool --nseeds 2`.

### 15.4 What the first survey shows (a = 0, h = 0.5, 72 × 72, default budget)

- Above θ ≈ 0.7–1 the lattice is paramagnetic at all b; the Polarized label grows out of the paramagnet with increasing b as a crossover.
- Cooling runs alone end in trapped textures: stripe–skyrmion mixtures at b ≲ 0.3 and skyrmion-rich states at every b in (0, 1), i.e. the trapping that Sec. 6 anticipates. The seeded histories are needed for the low-temperature diagram.
- With the energy rule, the low-θ row reproduces the classical sequence Helical → Skyrmion crystal → Polarized. The lattice crossings at θ = 0.02 fall between b = 0.20 and 0.25 (helix vs. 30-cell crystal; continuum b₁ ≈ 0.216) and near b ≈ 0.72 (continuum b_c ≈ 0.801 for the optimal density; the seed density is fixed, and h = 0.5 shifts lattice energies by ~6 %, e.g. the b = 0 helix gives ρ ≈ −0.47 instead of −0.5).
- Within the crystal region the cooled samples form a crystal (Q ≈ −25, sixfold peaks, ⟨|ψ6|⟩ ≈ 0.6–0.7) only below θ ≈ 0.1 and only near b ≈ 0.5; elsewhere they stay polycrystalline within the budget. The crystal–disordered boundary is therefore set by the ψ6 threshold and the sampling budget as much as by thermodynamics.

Known limitations, in the order they should be addressed: the Metropolis cooling budget (Sec. 6.2 asks for 10⁵ sweeps; `--neq/--nprod`), the fixed skyrmion count of the seed (`--skx-cells`, or several seed densities), the square-lattice anisotropy at h = 0.5 (the diagonal helix is 0.003 J per site lower than the axial one; use h = 0.25, Sec. 4.2), finite-size and sector checks (Sec. 8), and parallel tempering (Sec. 11.1), which is not implemented.

## References and related files

- Yu et al. (2010), [paper DOI](https://doi.org/10.1038/nature09124); [supplied supplement](theory_pdf/41586_2010_BFnature09124_MOESM321_ESM.pdf), especially pp. 4-6. Main source for the lattice model, two-dimensional reduction, Metropolis sweeps, and combined phase diagnostics.
- [phase_diagram_ba.jl](phase_diagram_ba.jl). Source for our continuum parameters and current fixed-radius energy comparison. Script SHA-256 at final review: `c40ffd7a068dc3d8f6761823cdc790b94b23047f46d21ba6c7d4482c8d9a0fe9`.
- Komatsu, Nonomura and Nishino (2021), Physical Review B **103**, 214404, [author manuscript](https://arxiv.org/html/2011.05496v3), [DOI](https://doi.org/10.1103/PhysRevB.103.214404). Relevant uniaxial-anisotropy model and structural diagnostics; Fig. 3(a) omits dipolar interactions.
- Metropolis et al. (1953), Journal of Chemical Physics **21**, 1087-1092, [original article](https://bayes.wustl.edu/Manual/EquationOfState.pdf), [DOI](https://doi.org/10.1063/1.1699114).
- Berg and Lüscher (1981), Nuclear Physics B **190**, 412-424, [DOI](https://doi.org/10.1016/0550-3213(81)90568-X). Geometric topological charge on a lattice.
- Hukushima and Nemoto (1996), Journal of the Physical Society of Japan **65**, 1604-1608, [author manuscript](https://arxiv.org/abs/cond-mat/9512035), [DOI](https://doi.org/10.1143/JPSJ.65.1604). Temperature exchange as an optional sampling extension.
