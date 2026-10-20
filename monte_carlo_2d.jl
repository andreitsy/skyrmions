#!/usr/bin/env julia
#=
monte_carlo_2d.jl — Metropolis Monte Carlo of the two-dimensional chiral magnet and its
(b, θ) phase diagram at fixed anisotropy a.

Method: README_monte_carlo_2d.md (adapted from the supplement to Yu et al., Nature 465, 901
(2010), pp. 4–6). "Sec." below refers to that document.

Model (Sec. 3) — classical unit spins on a periodic Nx × Ny square lattice,
    H = −J Σ_i (S_i·S_{i+x} + S_i·S_{i+y})
        − D Σ_i [x̂·(S_i × S_{i+x}) + ŷ·(S_i × S_{i+y})]
        − B Σ_i S_i^z + K Σ_i (1 − (S_i^z)²),
with J = 1, D = h, B = b h², K = a h²: h is the mesh spacing in units of ℓ_D = J_c/D_c, so the
continuum parameters b = B_c J_c/D_c² and a = K_c J_c/D_c² of phase_diagram_ba.jl are preserved
(Sec. 3.2). Temperature θ = k_B T / J_lat (Sec. 3.3).

Sampling (Sec. 5) — single-spin Metropolis with a mixed symmetric proposal (spherical cap of
adapted half-angle δ with probability 1 − pfull, uniform sphere with probability pfull) and the
exact local ΔH = −ΔS·h_i − K[(S'_z)² − S_z²]. One sweep (MCS) = N attempted updates, either in
checkerboard order (the two colours alternate; rows of one colour may be updated by several
threads, Sec. 12) or at random sites with replacement (--site-order random).

Histories (Sec. 6) — "cool" anneals random spins down the θ ladder; "heat-helix", "heat-skx"
and "heat-fm" heat a seeded helix / triple-q skyrmion lattice / polarized state up the ladder;
"field-up" / "field-down" sweep b at fixed θ. Each (history, seed, b) — or (history, seed, θ)
for field sweeps — is an independent trajectory with its own RNG; trajectories run in parallel
on Julia threads (start with -t auto).

Observables (Sec. 7–8) — energy components, m_z, |m_⊥|, c_V, χ_z, the Berg–Lüscher topological
charge Q and its map, the structure factors S(q) and S_zz(q) averaged over snapshots, Bragg-peak
analysis, skyrmion cores and the sixfold bond-orientational order ψ6, integrated autocorrelation
times, blocking errors and a drift check.

Classification (Sec. 9) — Paramagnet / Polarized / Helical / Skyrmion crystal /
Skyrmions (disordered) / Other ordered from |⟨Q⟩|, the Bragg peaks, their sixfold arrangement,
ψ6 and ⟨m_z⟩; "Unresolved" where histories or seeds disagree. The Polarized–Paramagnet
distinction is a crossover (threshold on ⟨m_z⟩), not a phase transition.

Quick start (repository root)
-----------------------------
  julia --project=scripts -e 'using Pkg; Pkg.instantiate()'       # once
  julia --project=scripts -t auto monte_carlo_2d.jl --selftest     # correctness checks (Sec. 13.2)
  julia --project=scripts -t auto monte_carlo_2d.jl                # default (b, θ) survey
  julia --project=scripts monte_carlo_2d.jl --help
=#

using LinearAlgebra
using Random
using Printf
using Dates
using Base.Threads

const NEED_PLOTS = !any(a -> a in ("-h", "--help", "--selftest", "--bench", "--no-plots"), ARGS)
if NEED_PLOTS
	using Plots
end

# Each trajectory runs on its own thread; the structure-factor matmuls must not spawn BLAS threads.
BLAS.set_num_threads(1)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
const HELP = """
monte_carlo_2d.jl — Metropolis Monte Carlo (b, θ) phase diagram of the 2D chiral magnet

Docs: README_monte_carlo_2d.md (method), README.md (overview)

USAGE
  julia --project=scripts -t auto monte_carlo_2d.jl [options]
  julia --project=scripts -t auto monte_carlo_2d.jl --selftest
  julia --project=scripts monte_carlo_2d.jl --help

MODEL (J_lat = 1, D_lat = h, B_lat = b h², K_lat = a h²; θ = k_B T / J_lat)
  --a VAL             easy-axis anisotropy a = K_c J_c/D_c²        (default 0)
  --h VAL             mesh spacing in units of ℓ_D = J_c/D_c       (default 0.5)
  --L N | --Nx --Ny   periodic lattice (even sizes for checkerboard) (default 72 × 72)

GRID
  --bmin/--bmax       field window                                  (default 0 … 1)
  --nb N | --db VAL   number of b points, or step                  (default 21)
  --b VAL             single field (nb = 1)
  --theta-max/-min    temperature ladder (descending)               (default 1.5 … 0.02)
  --ntheta N          ladder length                                 (default 24)
  --ladder geometric|linear                                         (default geometric)
  --theta VAL         single temperature (ntheta = 1)

HISTORIES (Sec. 6; comma separated)
  --histories LIST    cool | heat-helix | heat-skx | heat-fm | field-up | field-down
                      cool: random spins annealed down the ladder (the supplement's protocol);
                      heat-*: seeded helix / triple-q skyrmion lattice / polarized state heated up;
                      field-up/-down: b swept at each θ from a helix seed / polarized state
                                                                    (default cool,heat-helix,heat-skx,heat-fm)
  --nseeds N          independent seeds per history                 (default 1)
  --seed N            base seed                                     (default 12345)
  --skx-cells NX,MY   triple-q seed cells (default from h and box; N_sk = NX·MY)
  --helix-n N         helix seed winding (default: lowest lattice energy)

BUDGET (sweeps; 1 sweep = N attempted single-spin updates)
  --neq N             equilibration sweeps per point (incl. adaptation) (default 1000)
  --nprod N           production sweeps per point                    (default 1000)
  --nadapt N          sweeps of δ adaptation at the start of neq     (default 200)
  --final-factor F    multiply neq and nprod at the lowest θ         (default 1)
  --nquench N         T = 0 quench sweeps after the lowest θ         (default 500)
  --nmeas N           sweeps between energy/magnetization measurements (default 10)
  --nstruct N         sweeps between Q / structure-factor / core measurements (default 50)
  --site-order checkerboard|random                                  (default checkerboard)
  --inner-threads N|auto  threads per lattice (checkerboard)        (default auto)
  --pfull P           probability of the uniform-sphere proposal     (default 0.05)

CLASSIFICATION THRESHOLDS (Sec. 9)
  --peak-min VAL      min Bragg peak S(q)/N for modulated order      (default 0.02)
  --mz-pol VAL        ⟨m_z⟩ above which an unmodulated state is "Polarized" (default 0.7)
  --qmin VAL          min |⟨Q⟩| for skyrmions to be present          (default 3)
  --psi6-min VAL      min ⟨|ψ6|⟩ for a skyrmion crystal (if cores found) (default 0.5)
  --dom-frac VAL      peaks ≥ frac·P1 count as dominant              (default 0.3)
  --angle-tol DEG     tolerance on the 60°/90° peak arrangement      (default 15)
  --s-core VAL        core mask S_z < s-core                         (default 0)
  --qcore-min VAL     min |charge| of a masked region to be a core    (default 0.3)
  --resolve energy|strict  cells where histories/seeds disagree: label of the lowest-⟨E⟩ record
                      (marked with a dot; a low-T ground-state proxy, Sec. 10.1) or "Unresolved" (default energy)

OUTPUT
  --outdir DIR        parent directory (default outputs/monte_carlo_2d)
  --run-id NAME       run directory name (default a<a>_h<h>_L<L>_<timestamp>)
  --no-snapshots / --no-timeseries / --no-plots

OTHER
  --selftest          run the correctness checks of Sec. 13.2 and exit
  --bench             measure attempted updates per second and exit
  -h, --help

EXAMPLES
  # quick test of the whole pipeline (~30 s on 8 threads, 48², 11 × 12 grid, 300 + 300 sweeps)
  julia --project=scripts -t auto monte_carlo_2d.jl --L 48 --nb 11 --ntheta 12 --neq 300 --nprod 300 --nquench 100
  # default survey (72², 21 × 24 grid, 4 histories, 10³ + 10³ sweeps per point): ~5 min on 8 threads
  julia --project=scripts -t auto monte_carlo_2d.jl
  julia --project=scripts -t auto monte_carlo_2d.jl --histories cool --nseeds 2   # supplement protocol only
  julia --project=scripts -t auto monte_carlo_2d.jl --a 0.2 --db 0.1 --ntheta 12 --neq 500 --nprod 500
  julia --project=scripts -t auto monte_carlo_2d.jl --b 0.6 --theta 0.1 --neq 20000 --nprod 20000
  # cluster scale (Sec. 13.1 mesh; ≈ 6 × 10¹² attempts ≈ 1.5 h on 64 cores; --no-timeseries keeps the CSV small)
  julia --project=scripts -t \$SLURM_CPUS_PER_TASK monte_carlo_2d.jl --h 0.25 --L 144 --db 0.05 --ntheta 40 \\
        --neq 20000 --nprod 20000 --final-factor 5 --nstruct 200 --nseeds 2 --no-timeseries --run-id h0p25_L144

Cost ≈ (number of trajectories) × (points per trajectory) × (neq + nprod) × Nx × Ny attempts at ≈ 18 M
attempts/s per core; the script prints the total before starting and a rate at the end.
"""

function parse_args(args)
	o = Dict{String,Any}(
		"a" => 0.0, "h" => 0.5, "Nx" => 72, "Ny" => 72,
		"bmin" => 0.0, "bmax" => 1.0, "nb" => 21, "db" => nothing, "b" => nothing,
		"theta_max" => 1.5, "theta_min" => 0.02, "ntheta" => 24, "ladder" => "geometric", "theta" => nothing,
		"histories" => ["cool", "heat-helix", "heat-skx", "heat-fm"], "nseeds" => 1, "seed" => 12345,
		"skx_nx" => 0, "skx_my" => 0, "helix_n" => 0,
		"neq" => 1000, "nprod" => 1000, "nadapt" => 200, "final_factor" => 1, "nquench" => 500,
		"nmeas" => 10, "nstruct" => 50,
		"site_order" => "checkerboard", "inner" => 0, "pfull" => 0.05,
		"peak_min" => 0.02, "mz_pol" => 0.7, "qmin" => 3.0, "psi6_min" => 0.5,
		"dom_frac" => 0.3, "angle_tol" => 15.0, "s_core" => 0.0, "qcore_min" => 0.3, "resolve" => "energy",
		"outdir" => joinpath(@__DIR__, "outputs", "monte_carlo_2d"), "run_id" => "",
		"snapshots" => true, "timeseries" => true, "plots" => true,
		"selftest" => false, "bench" => false,
	)
	floats = Dict("--a"=>"a", "--h"=>"h", "--bmin"=>"bmin", "--bmax"=>"bmax", "--db"=>"db", "--b"=>"b",
		"--theta-max"=>"theta_max", "--theta-min"=>"theta_min", "--theta"=>"theta", "--pfull"=>"pfull",
		"--peak-min"=>"peak_min", "--mz-pol"=>"mz_pol", "--qmin"=>"qmin", "--psi6-min"=>"psi6_min",
		"--dom-frac"=>"dom_frac", "--angle-tol"=>"angle_tol", "--s-core"=>"s_core", "--qcore-min"=>"qcore_min")
	ints = Dict("--Nx"=>"Nx", "--Ny"=>"Ny", "--nb"=>"nb", "--ntheta"=>"ntheta", "--nseeds"=>"nseeds",
		"--seed"=>"seed", "--helix-n"=>"helix_n", "--neq"=>"neq", "--nprod"=>"nprod", "--nadapt"=>"nadapt",
		"--final-factor"=>"final_factor", "--nquench"=>"nquench", "--nmeas"=>"nmeas", "--nstruct"=>"nstruct")
	strs = Dict("--ladder"=>"ladder", "--site-order"=>"site_order", "--outdir"=>"outdir", "--run-id"=>"run_id",
		"--resolve"=>"resolve")
	i = 1
	while i ≤ length(args)
		arg = args[i]
		need(k) = (i < length(args) || error("$k needs a value"); args[i+=1])
		if haskey(floats, arg)
			o[floats[arg]] = parse(Float64, need(arg))
		elseif haskey(ints, arg)
			o[ints[arg]] = parse(Int, need(arg))
		elseif haskey(strs, arg)
			o[strs[arg]] = need(arg)
		elseif arg == "--L"
			L = parse(Int, need(arg)); o["Nx"] = L; o["Ny"] = L
		elseif arg == "--histories"
			o["histories"] = String.(split(need(arg), ','))
		elseif arg == "--skx-cells"
			v = parse.(Int, split(need(arg), ',')); length(v) == 2 || error("--skx-cells NX,MY")
			o["skx_nx"], o["skx_my"] = v
		elseif arg == "--inner-threads"
			v = need(arg); o["inner"] = v == "auto" ? 0 : parse(Int, v)
		elseif arg == "--no-snapshots"; o["snapshots"] = false
		elseif arg == "--no-timeseries"; o["timeseries"] = false
		elseif arg == "--no-plots"; o["plots"] = false
		elseif arg == "--selftest"; o["selftest"] = true
		elseif arg == "--bench"; o["bench"] = true
		elseif arg in ("-h", "--help"); print(HELP); exit(0)
		else
			error("unknown argument: $arg (use --help)")
		end
		i += 1
	end
	o["h"] > 0 || error("--h must be positive")
	o["theta_min"] > 0 || error("--theta-min must be positive (T = 0 is the quench, Sec. 5.3)")
	o["theta_max"] ≥ o["theta_min"] || error("--theta-max must be ≥ --theta-min")
	o["ladder"] in ("geometric", "linear") || error("--ladder must be geometric or linear")
	o["site_order"] in ("checkerboard", "random") || error("--site-order must be checkerboard or random")
	o["resolve"] in ("energy", "strict") || error("--resolve must be energy or strict")
	for hname in o["histories"]
		hname in ("cool", "heat-helix", "heat-skx", "heat-fm", "field-up", "field-down") ||
			error("unknown history $hname")
	end
	o["nmeas"] ≥ 1 && o["nstruct"] ≥ 1 || error("--nmeas and --nstruct must be ≥ 1")
	return o
end

function axis_grid(lo, hi; n=nothing, step=nothing)
	lo ≤ hi || error("axis range requires lo ≤ hi (got $lo > $hi)")
	if step !== nothing
		step > 0 || error("step must be positive")
		xs = collect(lo:step:hi)
		isempty(xs) && return [lo, hi]
		xs[end] < hi - 1e-9 * max(1.0, abs(hi)) && push!(xs, hi)
		return xs
	end
	n === nothing && error("axis_grid needs n or step")
	n == 1 && return [lo]
	return collect(range(lo, hi; length=n))
end

"""Descending temperature ladder (Sec. 6.2): geometric θ_{k+1} = r θ_k or linear."""
function temperature_ladder(θmax, θmin, n, kind)
	n == 1 && return [θmax]
	kind == "geometric" ? [θmax * (θmin/θmax)^((k-1)/(n-1)) for k in 1:n] :
	                      collect(range(θmax, θmin; length=n))
end

# ---------------------------------------------------------------------------
# Lattice model (Sec. 3)
# ---------------------------------------------------------------------------
struct Model
	Nx::Int; Ny::Int; N::Int
	J::Float64; D::Float64; B::Float64; K::Float64
	h::Float64; b::Float64; a::Float64
	ip::Vector{Int}; im::Vector{Int}; jp::Vector{Int}; jm::Vector{Int}
end

"""Lattice couplings from (h, b, a): D = J h, B = J b h², K = J a h² (Sec. 3.2). Override any coupling for tests."""
function Model(Nx::Int, Ny::Int; h::Float64, b::Float64=0.0, a::Float64=0.0, J::Float64=1.0,
               D::Float64=J*h, B::Float64=J*b*h^2, K::Float64=J*a*h^2)
	Model(Nx, Ny, Nx*Ny, J, D, B, K, h, b, a,
	      [mod1(i+1, Nx) for i in 1:Nx], [mod1(i-1, Nx) for i in 1:Nx],
	      [mod1(j+1, Ny) for j in 1:Ny], [mod1(j-1, Ny) for j in 1:Ny])
end

with_b(m::Model, b) = Model(m.Nx, m.Ny, m.N, m.J, m.D, m.J*b*m.h^2, m.K, m.h, b, m.a, m.ip, m.im, m.jp, m.jm)

"""Energy components (exchange, DMI, field, anisotropy), each bond counted once in +x, +y (Sec. 3.1)."""
function energy_components(S::Array{Float64,3}, m::Model)
	Eex = 0.0; Edm = 0.0; Eb = 0.0; Ek = 0.0
	@inbounds for j in 1:m.Ny
		jp = m.jp[j]
		for i in 1:m.Nx
			ip = m.ip[i]
			sx = S[1,i,j]; sy = S[2,i,j]; sz = S[3,i,j]
			tx = S[1,ip,j]; ty = S[2,ip,j]; tz = S[3,ip,j]      # S_{i+x}
			ux = S[1,i,jp]; uy = S[2,i,jp]; uz = S[3,i,jp]      # S_{i+y}
			Eex -= sx*tx + sy*ty + sz*tz + sx*ux + sy*uy + sz*uz
			Edm -= (sy*tz - sz*ty) + (sz*ux - sx*uz)             # x̂·(S×S_{+x}) + ŷ·(S×S_{+y})
			Eb -= sz
			Ek += 1.0 - sz*sz
		end
	end
	return m.J*Eex, m.D*Edm, m.B*Eb, m.K*Ek
end
total_energy(S, m::Model) = sum(energy_components(S, m))

"""Continuum-scaled energy density relative to the saturated state (Sec. 3.3): ρ = (H/N + 2J + B)/(J h²)."""
rho_continuum(e_site, m::Model) = (e_site + 2m.J + m.B)/(m.J*m.h^2)

"""Field h_i from the terms linear in S_i, Sec. 5.2 (four neighbours held fixed)."""
@inline function local_field(S::Array{Float64,3}, m::Model, i::Int, j::Int)
	@inbounds begin
		ip = m.ip[i]; im = m.im[i]; jp = m.jp[j]; jm = m.jm[j]
		axp = S[1,ip,j]; ayp = S[2,ip,j]; azp = S[3,ip,j]
		axm = S[1,im,j]; aym = S[2,im,j]; azm = S[3,im,j]
		bxp = S[1,i,jp]; byp = S[2,i,jp]; bzp = S[3,i,jp]
		bxm = S[1,i,jm]; bym = S[2,i,jm]; bzm = S[3,i,jm]
	end
	J = m.J; D = m.D
	# x̂ × (S_{i-x} − S_{i+x}) = (0, −Δz, Δy);  ŷ × (S_{i-y} − S_{i+y}) = (Δz, 0, −Δx)
	hx = J*(axp + axm + bxp + bxm) + D*(bzm - bzp)
	hy = J*(ayp + aym + byp + bym) - D*(azm - azp)
	hz = J*(azp + azm + bzp + bzm) + D*(aym - ayp) - D*(bxm - bxp) + m.B
	return hx, hy, hz
end

# ---------------------------------------------------------------------------
# Initial states (Sec. 6.1)
# ---------------------------------------------------------------------------
@inline function uniform_sphere(rng)
	z = 2.0*rand(rng) - 1.0
	r = sqrt(max(0.0, 1.0 - z*z))
	sφ, cφ = sincos(2π*rand(rng))
	return r*cφ, r*sφ, z
end

function random_spins!(S, rng)
	@inbounds for j in 1:size(S,3), i in 1:size(S,2)
		S[1,i,j], S[2,i,j], S[3,i,j] = uniform_sphere(rng)
	end
	S
end

function polarized_seed!(S, rng; ε=0.05)
	@inbounds for j in 1:size(S,3), i in 1:size(S,2)
		x = ε*randn(rng); y = ε*randn(rng); z = 1.0
		inv = 1/sqrt(x*x + y*y + z*z)
		S[1,i,j] = x*inv; S[2,i,j] = y*inv; S[3,i,j] = z*inv
	end
	S
end

"""Axial lattice helix energy per site, S = (0, −sin kx, cos kx): −J(1 + cos k) − D sin k."""
axial_helix_energy(m::Model, k) = -m.J*(1 + cos(k)) - m.D*sin(k)

"""Winding n of the axial helix seed with the lowest lattice energy in the box (Sec. 4.3)."""
best_helix_n(m::Model) = argmin(n -> axial_helix_energy(m, 2π*n/m.Nx), 1:max(1, m.Nx ÷ 2))

"""Helix seed S(x, y) = (0, −sin qx, cos qx), q = 2πn/Nx > 0 for D > 0 (Sec. 6.1)."""
function helix_seed!(S, m::Model; n::Int=0)
	n ≤ 0 && (n = best_helix_n(m))
	k = 2π*n/m.Nx
	@inbounds for j in 1:m.Ny, i in 1:m.Nx
		s, c = sincos(k*(i-1))
		S[1,i,j] = 0.0; S[2,i,j] = -s; S[3,i,j] = c
	end
	S
end

"""Diagonal lattice helix with wavevector (k, k) (Sec. 4.2), in-plane part along ẑ × q̂ with the helix sign."""
function diagonal_helix!(S, m::Model, n::Int)
	m.Nx == m.Ny || error("diagonal helix needs a square lattice")
	k = 2π*n/m.Nx
	@inbounds for j in 1:m.Ny, i in 1:m.Nx
		s, c = sincos(k*((i-1) + (j-1)))
		# ẑ × q̂ = (−1, 1)/√2 ; S_⊥ = −sin(q·r) (ẑ × q̂)
		S[1,i,j] = s/√2; S[2,i,j] = -s/√2; S[3,i,j] = c
	end
	S
end

"""Default triple-q cell counts (nx along x, my rows): |q| ≈ 1/ℓ_D, N_sk = nx·my (Sec. 6.1, 10.2)."""
function default_skx_cells(m::Model)
	nx = max(1, round(Int, m.Nx*m.h/(2π)))
	my = max(1, round(Int, (√3/2)*nx*m.Ny/m.Nx))
	nx, my
end

"""
Triple-q Bloch skyrmion-lattice seed: S ∝ m0 ẑ + Σ_k [−cos(q_k·r) ẑ + sin(q_k·r) (ẑ × q̂_k)], then normalized.
Each term is a helix of the D > 0 helicity shifted by π, so cores (S_z = −1) sit on a triangular lattice
with nx·my cells in the periodic box; all three q_k are commensurate.
"""
function skx_seed!(S, m::Model; nx::Int=0, my::Int=0, m0=1.0)
	if nx ≤ 0 || my ≤ 0
		nx, my = default_skx_cells(m)
	end
	k2x = -round(Int, nx/2)
	qs = ((2π*nx/m.Nx, 0.0), (2π*k2x/m.Nx, 2π*my/m.Ny), (2π*(-nx - k2x)/m.Nx, -2π*my/m.Ny))
	@inbounds for j in 1:m.Ny, i in 1:m.Nx
		x = i - 1; y = j - 1
		sx = 0.0; sy = 0.0; sz = m0
		for (qx, qy) in qs
			qn = hypot(qx, qy)
			s, c = sincos(qx*x + qy*y)
			sz -= c
			sx += s*(-qy/qn); sy += s*(qx/qn)        # ẑ × q̂ = (−q̂_y, q̂_x)
		end
		inv = 1/sqrt(sx*sx + sy*sy + sz*sz)
		S[1,i,j] = sx*inv; S[2,i,j] = sy*inv; S[3,i,j] = sz*inv
	end
	S
end

"""Single compact Bloch skyrmion (Sec. 6.1): β(r) = π(1 − r/R0) for r < R0, γ = π/2."""
function bloch_seed!(S, m::Model; R0=nothing, γ=π/2, center=(m.Nx/2 + 0.5, m.Ny/2 + 0.5))
	R0 === nothing && (R0 = min(m.Nx, m.Ny)/3)
	@inbounds for j in 1:m.Ny, i in 1:m.Nx
		dx = i - center[1]; dy = j - center[2]
		r = hypot(dx, dy)
		β = r < R0 ? π*(1 - r/R0) : 0.0
		φ = atan(dy, dx)
		sβ, cβ = sincos(β)
		S[1,i,j] = sβ*cos(φ + γ); S[2,i,j] = sβ*sin(φ + γ); S[3,i,j] = cβ
	end
	S
end

# ---------------------------------------------------------------------------
# Metropolis update (Sec. 5)
# ---------------------------------------------------------------------------
"""Symmetric proposal: uniform sphere with probability pfull, else uniform in a cap of half-angle δ (Sec. 5.1)."""
@inline function propose_spin(rng::Xoshiro, sx, sy, sz, cosδ, pfull)
	rand(rng) < pfull && return uniform_sphere(rng)
	# orthonormal frame (e1, e2) ⊥ S from the lattice axis least aligned with S
	if abs(sx) < 0.7
		ex = 0.0; ey = -sz; ez = sy            # x̂ × S
	else
		ex = sz; ey = 0.0; ez = -sx            # ŷ × S
	end
	inv = 1.0/sqrt(ex*ex + ey*ey + ez*ez)
	ex *= inv; ey *= inv; ez *= inv
	fx = sy*ez - sz*ey; fy = sz*ex - sx*ez; fz = sx*ey - sy*ex   # e2 = S × e1
	c = 1.0 - rand(rng)*(1.0 - cosδ)
	s = sqrt(max(0.0, 1.0 - c*c))
	sφ, cφ = sincos(2π*rand(rng))
	nx = c*sx + s*(cφ*ex + sφ*fx)
	ny = c*sy + s*(cφ*ey + sφ*fy)
	nz = c*sz + s*(cφ*ez + sφ*fz)
	inv = 1.0/sqrt(nx*nx + ny*ny + nz*nz)
	return nx*inv, ny*inv, nz*inv
end

"""One attempted update at site (i, j). β = 1/θ (β = Inf gives the T = 0 quench). Returns (ΔH, accepted)."""
@inline function update_site!(S::Array{Float64,3}, m::Model, i::Int, j::Int, rng::Xoshiro,
                              β::Float64, cosδ::Float64, pfull::Float64)
	@inbounds begin
		sx = S[1,i,j]; sy = S[2,i,j]; sz = S[3,i,j]
		hx, hy, hz = local_field(S, m, i, j)
		nx, ny, nz = propose_spin(rng, sx, sy, sz, cosδ, pfull)
		ΔH = -((nx - sx)*hx + (ny - sy)*hy + (nz - sz)*hz) - m.K*(nz*nz - sz*sz)   # Sec. 5.2
		if ΔH <= 0.0 || log(rand(rng)) < -ΔH*β                                      # Sec. 5.3
			S[1,i,j] = nx; S[2,i,j] = ny; S[3,i,j] = nz
			return ΔH, 1
		end
	end
	return 0.0, 0
end

"""Update every site of one checkerboard colour in rows jrange (lattice order)."""
function sweep_colour!(S, m::Model, rng::Xoshiro, β, cosδ, pfull, colour::Int, jrange::UnitRange{Int})
	dE = 0.0; nacc = 0
	@inbounds for j in jrange
		istart = ((colour + j) & 1) == 1 ? 1 : 2
		for i in istart:2:m.Nx
			d, acc = update_site!(S, m, i, j, rng, β, cosδ, pfull)
			dE += d; nacc += acc
		end
	end
	return dE, nacc
end

mutable struct Sampler
	rngs::Vector{Xoshiro}          # one private stream per row block
	blocks::Vector{UnitRange{Int}} # row blocks updated concurrently (checkerboard)
	order::Symbol                  # :checkerboard or :random
	δ::Float64                     # cap half-angle
	pfull::Float64
	E::Float64                     # cached total energy
	nacc::Int
	natt::Int
end

function row_blocks(Ny, nblocks)
	nblocks = clamp(nblocks, 1, Ny)
	edges = round.(Int, range(0, Ny; length=nblocks+1))
	[edges[k]+1:edges[k+1] for k in 1:nblocks if edges[k+1] > edges[k]]
end

"""One Monte Carlo sweep = N attempted updates (Sec. 5.3); updates the cached energy and acceptance counters."""
function sweep!(S, m::Model, sp::Sampler, β::Float64)
	cosδ = cos(sp.δ)
	if sp.order == :random
		rng = sp.rngs[1]; dE = 0.0; nacc = 0
		@inbounds for _ in 1:m.N
			i = rand(rng, 1:m.Nx); j = rand(rng, 1:m.Ny)
			d, acc = update_site!(S, m, i, j, rng, β, cosδ, sp.pfull)
			dE += d; nacc += acc
		end
		sp.E += dE; sp.nacc += nacc
	elseif length(sp.blocks) == 1
		for colour in 0:1
			dE, nacc = sweep_colour!(S, m, sp.rngs[1], β, cosδ, sp.pfull, colour, 1:m.Ny)
			sp.E += dE; sp.nacc += nacc
		end
	else
		nb = length(sp.blocks)
		res = Vector{Tuple{Float64,Int}}(undef, nb)
		for colour in 0:1
			@sync for k in 1:nb
				Threads.@spawn begin
					res[k] = sweep_colour!(S, m, sp.rngs[k], β, cosδ, sp.pfull, colour, sp.blocks[k])
				end
			end
			for k in 1:nb
				sp.E += res[k][1]; sp.nacc += res[k][2]
			end
		end
	end
	sp.natt += m.N
	return nothing
end

"""Adjust δ towards an acceptance window during the adaptation stage only (Sec. 5.1)."""
function adapt_delta!(sp::Sampler, acc; lo=0.3, hi=0.6)
	if acc > hi
		sp.δ = min(sp.δ*1.2, π)
	elseif acc < lo
		sp.δ = max(sp.δ/1.2, 1e-4)
	end
	sp
end

"""T = 0 Metropolis quench (accept only ΔH ≤ 0) with a shrinking cap, Sec. 10.1 ground-state comparison."""
function quench!(S, m::Model, sp::Sampler, nsweeps)
	sp.δ = min(sp.δ, 0.5)
	nbatch = 10
	for _ in 1:max(1, nsweeps ÷ nbatch)
		sp.nacc = 0; sp.natt = 0
		for _ in 1:nbatch
			sweep!(S, m, sp, Inf)
		end
		acc = sp.nacc / sp.natt
		acc < 0.3 && (sp.δ = max(sp.δ/1.3, 1e-5))
		acc > 0.6 && (sp.δ = min(sp.δ*1.3, π))
	end
	sp.E = total_energy(S, m)
	sp.E
end

# ---------------------------------------------------------------------------
# Observables (Sec. 7)
# ---------------------------------------------------------------------------
function magnetization(S, m::Model)
	mx = 0.0; my = 0.0; mz = 0.0; sz2 = 0.0
	@inbounds for j in 1:m.Ny, i in 1:m.Nx
		mx += S[1,i,j]; my += S[2,i,j]; mz += S[3,i,j]; sz2 += S[3,i,j]^2
	end
	return mx/m.N, my/m.N, mz/m.N, sz2/m.N
end

"""Oriented spherical area of the triangle (u, v, w) (Berg–Lüscher, Sec. 7.2)."""
@inline function solid_angle(ux, uy, uz, vx, vy, vz, wx, wy, wz)
	num = ux*(vy*wz - vz*wy) + uy*(vz*wx - vx*wz) + uz*(vx*wy - vy*wx)
	den = 1.0 + (ux*vx + uy*vy + uz*vz) + (vx*wx + vy*wy + vz*wz) + (wx*ux + wy*uy + wz*uz)
	return 2*atan(num, den), den
end

"""
Lattice topological charge Q = Σ Ω/4π over triangles (A,B,C), (A,C,D) of every plaquette, including
those across the periodic boundary (Sec. 7.2). Fills qmap[i, j] with the plaquette charge whose lower-left
corner is (i, j); returns (Q, number of near-degenerate triangles with denominator < 0.1).
"""
function charge_map!(qmap::Matrix{Float64}, S, m::Model)
	Q = 0.0; ndeg = 0
	@inbounds for j in 1:m.Ny
		jp = m.jp[j]
		for i in 1:m.Nx
			ip = m.ip[i]
			ax = S[1,i,j];  ay = S[2,i,j];  az = S[3,i,j]
			bx = S[1,ip,j]; by = S[2,ip,j]; bz = S[3,ip,j]
			cx = S[1,ip,jp]; cy = S[2,ip,jp]; cz = S[3,ip,jp]
			dx = S[1,i,jp]; dy = S[2,i,jp]; dz = S[3,i,jp]
			Ω1, d1 = solid_angle(ax, ay, az, bx, by, bz, cx, cy, cz)
			Ω2, d2 = solid_angle(ax, ay, az, cx, cy, cz, dx, dy, dz)
			ndeg += (d1 < 0.1) + (d2 < 0.1)
			q = (Ω1 + Ω2)/(4π)
			qmap[i,j] = q
			Q += q
		end
	end
	return Q, ndeg
end

"""Structure-factor work space: DFT matrices and complex buffers (no FFT dependency; two matmuls per component)."""
struct SFWork
	Wx::Matrix{ComplexF64}
	WyT::Matrix{ComplexF64}
	A::Matrix{ComplexF64}
	T::Matrix{ComplexF64}
	F::Matrix{ComplexF64}
end
SFWork(Nx, Ny) = SFWork([cis(-2π*(k-1)*(x-1)/Nx) for k in 1:Nx, x in 1:Nx],
                        [cis(-2π*(y-1)*(k-1)/Ny) for y in 1:Ny, k in 1:Ny],
                        zeros(ComplexF64, Nx, Ny), zeros(ComplexF64, Nx, Ny), zeros(ComplexF64, Nx, Ny))

"""
Accumulate S(q)/N = |Σ_j (S_j − S̄) e^{−iq·x_j}|² / N² summed over components (Sq) and for S_z alone (Szz),
Sec. 7.3. Each snapshot's own mean is subtracted; Parseval: Σ_q S(q)/N = ⟨|S − S̄|²⟩ ≤ 1, a Bragg peak is O(1).
"""
function accumulate_sf!(Sq::Matrix{Float64}, Szz::Matrix{Float64}, S, m::Model, w::SFWork)
	N2 = float(m.N)^2
	for c in 1:3
		μ = 0.0
		@inbounds for j in 1:m.Ny, i in 1:m.Nx
			μ += S[c,i,j]
		end
		μ /= m.N
		@inbounds for j in 1:m.Ny, i in 1:m.Nx
			w.A[i,j] = S[c,i,j] - μ
		end
		mul!(w.T, w.Wx, w.A)
		mul!(w.F, w.T, w.WyT)
		@inbounds for j in 1:m.Ny, i in 1:m.Nx
			v = abs2(w.F[i,j])/N2
			Sq[i,j] += v
			c == 3 && (Szz[i,j] += v)
		end
	end
	nothing
end

signed_k(i, N) = i - 1 ≤ N ÷ 2 ? i - 1 : i - 1 - N

struct PeakInfo
	P1::Float64; P2::Float64; P3::Float64
	q1x::Float64; q1y::Float64; q1::Float64   # radians per lattice site
	ndom::Int
	sixfold::Bool
	square::Bool
end

"""
Bragg-peak analysis of a (time-averaged) S(q)/N map: local maxima over the periodic 8-neighbourhood, one
representative per ±q pair, dominant pairs = those ≥ dom_frac·P1 within [0.6, 1.4]·|q1|. Sixfold: three
dominant pairs ~60° apart; square: two dominant pairs ~90° apart (Sec. 7.3, 9).
"""
function analyze_peaks(A::AbstractMatrix, Nx, Ny; dom_frac=0.3, angle_tol=deg2rad(15))
	peaks = Tuple{Float64,Int,Int}[]
	@inbounds for j in 1:Ny, i in 1:Nx
		(i == 1 && j == 1) && continue
		kx = signed_k(i, Nx); ky = signed_k(j, Ny)
		(ky > 0 || (ky == 0 && kx > 0)) || continue       # one representative per ±q pair
		v = A[i,j]
		ismax = true
		for dj in -1:1, di in -1:1
			(di == 0 && dj == 0) && continue
			ii = mod1(i + di, Nx); jj = mod1(j + dj, Ny)
			u = A[ii,jj]
			if u > v || (u == v && (jj, ii) < (j, i))
				ismax = false; break
			end
		end
		ismax && push!(peaks, (v, kx, ky))
	end
	isempty(peaks) && return PeakInfo(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, false, false)
	sort!(peaks; by=first, rev=true)
	P1, kx1, ky1 = peaks[1]
	q1x = 2π*kx1/Nx; q1y = 2π*ky1/Ny; q1 = hypot(q1x, q1y)
	dom = Tuple{Float64,Float64,Float64}[]
	for (v, kx, ky) in peaks
		v ≥ dom_frac*P1 || break
		qx = 2π*kx/Nx; qy = 2π*ky/Ny; qn = hypot(qx, qy)
		0.6q1 ≤ qn ≤ 1.4q1 && push!(dom, (v, qx, qy))
	end
	P2 = length(peaks) ≥ 2 ? peaks[2][1] : 0.0
	P3 = length(peaks) ≥ 3 ? peaks[3][1] : 0.0
	angles = [mod(atan(qy, qx), π) for (_, qx, qy) in dom]
	function pair_angle(α, β)
		d = abs(α - β); min(d, π - d)
	end
	sixfold = false; square = false
	if length(dom) ≥ 3
		d12 = pair_angle(angles[1], angles[2]); d13 = pair_angle(angles[1], angles[3]); d23 = pair_angle(angles[2], angles[3])
		sixfold = all(abs(d - π/3) ≤ angle_tol for d in (d12, d13, d23))
	end
	if length(dom) == 2
		square = abs(pair_angle(angles[1], angles[2]) - π/2) ≤ angle_tol
	end
	PeakInfo(P1, P2, P3, q1x, q1y, q1, length(dom), sixfold, square)
end

"""Per-trajectory measurement buffers."""
struct Work
	qmap::Matrix{Float64}
	region::Matrix{Int32}
	pstamp::Matrix{Int32}
	stack::Vector{Int}
	sf::SFWork
	Sq::Matrix{Float64}
	Szz::Matrix{Float64}
	centres::Vector{Tuple{Float64,Float64}}
end
Work(Nx, Ny) = Work(zeros(Nx, Ny), zeros(Int32, Nx, Ny), zeros(Int32, Nx, Ny), Int[], SFWork(Nx, Ny),
                    zeros(Nx, Ny), zeros(Nx, Ny), Tuple{Float64,Float64}[])

"""
Skyrmion cores (Sec. 7.4): connected regions of S_z < s_core (periodic), kept when their area ≥ min_area and
the charge of the plaquettes touching them satisfies |Q_region| ≥ qcore_min (a core bounded by the S_z = 0
contour carries |Q| ≈ 1/2; a stripe segment carries ≈ 0). Centres are periodic circular means weighted by
(s_core − S_z). Requires charge_map! to have filled w.qmap. Returns the number of cores (centres in w.centres).
"""
function detect_cores!(w::Work, S, m::Model; s_core=0.0, qcore_min=0.3, min_area=2)
	Nx, Ny = m.Nx, m.Ny
	fill!(w.region, 0); fill!(w.pstamp, 0); empty!(w.centres)
	nreg = Int32(0)
	@inbounds for j in 1:Ny, i in 1:Nx
		(S[3,i,j] < s_core && w.region[i,j] == 0) || continue
		nreg += 1
		empty!(w.stack); push!(w.stack, i + (j-1)*Nx); w.region[i,j] = nreg
		area = 0; qsum = 0.0; cxs = 0.0; cxc = 0.0; cys = 0.0; cyc = 0.0
		while !isempty(w.stack)
			idx = pop!(w.stack)
			ii = (idx - 1) % Nx + 1; jj = (idx - 1) ÷ Nx + 1
			area += 1
			wgt = s_core - S[3,ii,jj]
			sx_, cx_ = sincos(2π*(ii-1)/Nx); sy_, cy_ = sincos(2π*(jj-1)/Ny)
			cxs += wgt*sx_; cxc += wgt*cx_; cys += wgt*sy_; cyc += wgt*cy_
			im = m.im[ii]; jm = m.jm[jj]; ip = m.ip[ii]; jp = m.jp[jj]
			# plaquettes touching this site (lower-left corners)
			for (pi, pj) in ((ii, jj), (im, jj), (ii, jm), (im, jm))
				if w.pstamp[pi,pj] != nreg
					w.pstamp[pi,pj] = nreg; qsum += w.qmap[pi,pj]
				end
			end
			for (ni, nj) in ((ip, jj), (im, jj), (ii, jp), (ii, jm))
				if S[3,ni,nj] < s_core && w.region[ni,nj] == 0
					w.region[ni,nj] = nreg; push!(w.stack, ni + (nj-1)*Nx)
				end
			end
		end
		if area ≥ min_area && abs(qsum) ≥ qcore_min
			cx = mod(atan(cxs, cxc)*Nx/(2π), Nx) + 1
			cy = mod(atan(cys, cyc)*Ny/(2π), Ny) + 1
			push!(w.centres, (cx, cy))
		end
	end
	return length(w.centres)
end

function median_of(x)
	isempty(x) && return NaN
	s = sort(x); n = length(s)
	isodd(n) ? s[(n+1)÷2] : (s[n÷2] + s[n÷2+1])/2
end

"""
Sixfold bond-orientational order ψ6 = (1/N_sk) Σ_j (1/z_j) Σ_k e^{6iφ_jk} with periodic neighbours within
rfac × median nearest-neighbour distance (Sec. 7.4). Returns (|ψ6| global, mean local |ψ6,j|); NaN if < 3 cores.
"""
function psi6_order(centres, Nx, Ny; rfac=1.35)
	n = length(centres)
	n < 3 && return NaN, NaN
	dx = zeros(n, n); dy = zeros(n, n); d = zeros(n, n)
	for a in 1:n, b in 1:n
		ddx = centres[b][1] - centres[a][1]; ddx -= Nx*round(ddx/Nx)
		ddy = centres[b][2] - centres[a][2]; ddy -= Ny*round(ddy/Ny)
		dx[a,b] = ddx; dy[a,b] = ddy; d[a,b] = hypot(ddx, ddy)
	end
	rnn = [minimum(d[a,b] for b in 1:n if b != a) for a in 1:n]
	rc = rfac*median_of(rnn)
	gsum = 0.0im; lsum = 0.0; cnt = 0
	for a in 1:n
		ψ = 0.0im; z = 0
		for b in 1:n
			(b != a && d[a,b] ≤ rc) || continue
			ψ += cis(6*atan(dy[a,b], dx[a,b])); z += 1
		end
		z == 0 && continue
		ψ /= z; gsum += ψ; lsum += abs(ψ); cnt += 1
	end
	cnt == 0 && return NaN, NaN
	return abs(gsum)/cnt, lsum/cnt
end

# ---------------------------------------------------------------------------
# Statistics (Sec. 8)
# ---------------------------------------------------------------------------
mean_of(x) = isempty(x) ? NaN : sum(x)/length(x)
function var_of(x)
	isempty(x) && return NaN
	μ = mean_of(x)
	sum(abs2, x .- μ)/length(x)
end

"""Integrated autocorrelation time (measurement intervals): τ = 1/2 + Σ C(t), truncated at the first C ≤ 0 or t ≥ 6τ."""
function tau_int(x::AbstractVector{Float64})
	M = length(x)
	M < 8 && return 0.5
	μ = mean_of(x); v = var_of(x)
	v ≤ 1e-300 && return 0.5
	τ = 0.5
	for t in 1:(M ÷ 2)
		c = 0.0
		@inbounds for s in 1:(M - t)
			c += (x[s] - μ)*(x[s+t] - μ)
		end
		c /= (M - t)*v
		c ≤ 0 && break
		τ += c
		t ≥ 6τ && break
	end
	τ
end

"""Blocking error of the mean with blocks longer than 2τ (≥ 4 blocks), else the τ-corrected naive error."""
function block_error(x::AbstractVector{Float64}, τ)
	M = length(x)
	M < 4 && return NaN
	blen = max(ceil(Int, 2τ), M ÷ 16, 1)
	nb = M ÷ blen
	if nb ≥ 4
		bm = [mean_of(view(x, (k-1)*blen+1:k*blen)) for k in 1:nb]
		return sqrt(var_of(bm)/(nb - 1))
	end
	return sqrt(2τ*var_of(x)/M)
end

"""Mean, error, τ, N_eff and a drift flag (first and second half agree within 3σ)."""
function series_stats(x::AbstractVector{Float64})
	M = length(x)
	M == 0 && return (mean=NaN, err=NaN, tau=NaN, neff=NaN, drift_ok=false)
	μ = mean_of(x); τ = tau_int(x); err = block_error(x, τ)
	drift_ok = true
	if M ≥ 8
		h = M ÷ 2
		x1 = view(x, 1:h); x2 = view(x, h+1:M)
		e1 = sqrt(2tau_int(x1)*var_of(x1)/h); e2 = sqrt(2tau_int(x2)*var_of(x2)/(M - h))
		drift_ok = abs(mean_of(x1) - mean_of(x2)) ≤ 3*sqrt(e1^2 + e2^2) + 1e-12
	end
	(mean=μ, err=err, tau=τ, neff=M/(2τ), drift_ok=drift_ok)
end

# ---------------------------------------------------------------------------
# Phase labels (Sec. 9)
# ---------------------------------------------------------------------------
const LABELS = ("Paramagnet", "Polarized", "Helical", "Skyrmion crystal", "Skyrmions (disordered)", "Unresolved")
const LABEL_CHARS = ('P', 'F', 'H', 'S', 's', '?')
const LABEL_RGB = ((0.98, 0.96, 0.82), (0.86, 0.86, 0.86), (0.30, 0.55, 0.85), (0.85, 0.20, 0.25),
                   (0.95, 0.65, 0.20), (0.40, 0.40, 0.40))
const NLABELS = length(LABELS)
const UNRESOLVED = NLABELS
label_id(s::AbstractString) = something(findfirst(==(s), LABELS))
"""Classes used for the agreement test: {Paramagnet, Polarized} share one class (crossover, not a transition)."""
phase_class(id::Int) = id ≤ 2 ? 1 : id == 3 ? 2 : id ≤ 5 ? 3 : 4

struct Thresholds
	peak_min::Float64; mz_pol::Float64; qmin::Float64; psi6_min::Float64
	dom_frac::Float64; angle_tol::Float64; s_core::Float64; qcore_min::Float64
end

"""
Label one (history, b, θ) record from its diagnostics (Sec. 9 table).

Skyrmions are "present" when the net charge is large, statistically significant and persistent over the
production run (std(Q) ≤ |⟨Q⟩|/2; thermal charge fluctuations of a paramagnet are large and short-lived).
A crystal additionally needs a Bragg peak with three dominant pairs ~60° apart and ⟨|ψ6|⟩ ≥ psi6_min
(when cores were found). Modulated states without persistent charge are helical (single- or multi-domain
stripes; distorted helicoids have harmonics outside the dominant window). Otherwise the state is
unmodulated: Polarized (⟨m_z⟩ ≥ mz_pol) or Paramagnet — a crossover, not a transition.
"""
function classify(Qmean, Qerr, Qstd, peaks::PeakInfo, psi6, mz, th::Thresholds)
	skyrmions = abs(Qmean) ≥ th.qmin && abs(Qmean) > 3*max(Qerr, 0.0) && Qstd ≤ 0.5*abs(Qmean)
	modulated = peaks.P1 ≥ th.peak_min
	if skyrmions
		if modulated && peaks.sixfold && (isnan(psi6) || psi6 ≥ th.psi6_min)
			return "Skyrmion crystal"
		else
			return "Skyrmions (disordered)"
		end
	elseif modulated
		return "Helical"
	else
		return mz ≥ th.mz_pol ? "Polarized" : "Paramagnet"
	end
end

# ---------------------------------------------------------------------------
# Trajectories (Sec. 6, 12)
# ---------------------------------------------------------------------------
struct Trajectory
	hist::String
	seed::Int
	init::Symbol                    # :random, :helix, :skx, :polarized
	ladder::Symbol                  # :temperature or :field
	points::Vector{Tuple{Float64,Float64}}  # (b, θ) in visiting order
	bidx::Vector{Int}
	tidx::Vector{Int}
end

function build_trajectories(hists, nseeds, bs, θs)
	nb = length(bs); nθ = length(θs)
	trajs = Trajectory[]
	for hist in hists, seed in 1:nseeds
		if hist == "cool"
			for j in 1:nb
				push!(trajs, Trajectory(hist, seed, :random, :temperature,
					[(bs[j], θs[k]) for k in 1:nθ], fill(j, nθ), collect(1:nθ)))
			end
		elseif startswith(hist, "heat-")
			init = hist == "heat-helix" ? :helix : hist == "heat-skx" ? :skx : :polarized
			for j in 1:nb
				push!(trajs, Trajectory(hist, seed, init, :temperature,
					[(bs[j], θs[k]) for k in nθ:-1:1], fill(j, nθ), collect(nθ:-1:1)))
			end
		elseif hist == "field-up"
			for k in 1:nθ
				push!(trajs, Trajectory(hist, seed, :helix, :field,
					[(bs[j], θs[k]) for j in 1:nb], collect(1:nb), fill(k, nb)))
			end
		elseif hist == "field-down"
			for k in 1:nθ
				push!(trajs, Trajectory(hist, seed, :polarized, :field,
					[(bs[j], θs[k]) for j in nb:-1:1], collect(nb:-1:1), fill(k, nb)))
			end
		end
	end
	trajs
end

Base.@kwdef struct TResult
	hist::String; seed::Int; b::Float64; θ::Float64; bidx::Int; tidx::Int
	e::Float64; e_err::Float64; tau_e::Float64; neff_e::Float64
	e_ex::Float64; e_dm::Float64; e_b::Float64; e_k::Float64
	cV::Float64
	mz::Float64; mz_err::Float64; tau_mz::Float64; chi::Float64; mperp::Float64; sz2::Float64
	Q::Float64; Q_err::Float64; Q_std::Float64; tau_Q::Float64; neff_Q::Float64; absQ::Float64
	nsk::Float64; psi6::Float64; psi6_local::Float64; R_eff::Float64
	P1::Float64; P2::Float64; P3::Float64; q1x::Float64; q1y::Float64; qnorm::Float64
	λ_sites::Float64; λ_lD::Float64; ndom::Int; sixfold::Bool; square::Bool
	acc::Float64; δ::Float64; echeck::Float64; degen::Float64
	drift_ok::Bool
	ρ::Float64; ρ_quench::Float64; Q_quench::Float64
	label::String
	Sz::Matrix{Float32}
	Sq::Matrix{Float32}
	Sfull::Array{Float64,3}
	ts_sweep::Vector{Int}; ts_e::Vector{Float64}; ts_mz::Vector{Float64}
	ts_Qsweep::Vector{Int}; ts_Q::Vector{Float64}
end

const PRINT_LOCK = ReentrantLock()

function init_spins!(S, tr::Trajectory, m::Model, rng, cfg)
	if tr.init == :random
		random_spins!(S, rng)
	elseif tr.init == :helix
		helix_seed!(S, m; n=cfg.helix_n)
	elseif tr.init == :skx
		skx_seed!(S, m; nx=cfg.skx_nx, my=cfg.skx_my)
	else
		polarized_seed!(S, rng)
	end
end

"""Run one trajectory through its (b, θ) points; returns one TResult per point."""
function run_trajectory(tr::Trajectory, cfg, inner::Int, th::Thresholds)
	Nx, Ny = cfg.Nx, cfg.Ny
	m = Model(Nx, Ny; h=cfg.h, b=tr.points[1][1], a=cfg.a)
	S = Array{Float64}(undef, 3, Nx, Ny)
	key = (cfg.seed, tr.hist, tr.seed, tr.bidx[1], tr.tidx[1])
	rng0 = Xoshiro(hash(key))
	init_spins!(S, tr, m, rng0, cfg)
	order = cfg.site_order == "random" ? :random : :checkerboard
	nblocks = order == :random ? 1 : inner
	if nblocks > 1 && (isodd(Nx) || isodd(Ny))
		nblocks = 1   # the parallel checkerboard needs even periodic dimensions (Sec. 12)
	end
	blocks = row_blocks(Ny, nblocks)
	rngs = [Xoshiro(hash((key, k))) for k in 1:length(blocks)]
	sp = Sampler(rngs, blocks, order, 1.0, cfg.pfull, total_energy(S, m), 0, 0)
	w = Work(Nx, Ny)
	θmin = minimum(last, tr.points)
	out = TResult[]
	for (p, (b, θ)) in enumerate(tr.points)
		if b != m.b
			m = with_b(m, b); sp.E = total_energy(S, m)
		end
		β = 1/θ
		is_tmin = tr.ladder == :temperature && θ == θmin
		fac = is_tmin ? cfg.final_factor : 1
		neq = cfg.neq*fac; nprod = cfg.nprod*fac
		# --- adaptation of δ, then frozen (Sec. 5.1) ---
		nbatch = 10
		nadapt = min(cfg.nadapt, neq)
		done_sw = 0
		while done_sw < nadapt
			sp.nacc = 0; sp.natt = 0
			nsw = min(nbatch, nadapt - done_sw)
			for _ in 1:nsw
				sweep!(S, m, sp, β)
			end
			adapt_delta!(sp, sp.nacc/sp.natt)
			done_sw += nsw
		end
		# --- equilibration (Sec. 6) ---
		for _ in 1:(neq - nadapt)
			sweep!(S, m, sp, β)
		end
		Efull = total_energy(S, m)
		echeck = abs(sp.E - Efull)/m.N; sp.E = Efull     # cached vs full energy (Sec. 12)
		# --- production (Sec. 7) ---
		sp.nacc = 0; sp.natt = 0
		fill!(w.Sq, 0.0); fill!(w.Szz, 0.0); nsq = 0
		ts_sweep = Int[]; ts_e = Float64[]; ts_ex = Float64[]; ts_dm = Float64[]; ts_b = Float64[]; ts_k = Float64[]
		ts_mz = Float64[]; ts_mperp = Float64[]; ts_sz2 = Float64[]
		ts_Qsweep = Int[]; ts_Q = Float64[]; ts_nsk = Float64[]; ts_psi6 = Float64[]; ts_psi6l = Float64[]; ndeg_tot = 0
		nstruct = min(cfg.nstruct, nprod); nmeas = min(cfg.nmeas, nprod)
		for s in 1:nprod
			sweep!(S, m, sp, β)
			if s % nmeas == 0
				Eex, Edm, Eb, Ek = energy_components(S, m)
				push!(ts_sweep, s); push!(ts_e, (Eex + Edm + Eb + Ek)/m.N)
				push!(ts_ex, Eex/m.N); push!(ts_dm, Edm/m.N); push!(ts_b, Eb/m.N); push!(ts_k, Ek/m.N)
				mx, my, mz, sz2 = magnetization(S, m)
				push!(ts_mz, mz); push!(ts_mperp, hypot(mx, my)); push!(ts_sz2, sz2)
			end
			if s % nstruct == 0
				Q, ndeg = charge_map!(w.qmap, S, m)
				ndeg_tot += ndeg
				push!(ts_Qsweep, s); push!(ts_Q, Q)
				accumulate_sf!(w.Sq, w.Szz, S, m, w.sf); nsq += 1
				ncore = detect_cores!(w, S, m; s_core=th.s_core, qcore_min=th.qcore_min)
				push!(ts_nsk, ncore)
				g, l = psi6_order(w.centres, Nx, Ny)
				push!(ts_psi6, g); push!(ts_psi6l, l)
			end
		end
		# --- reduce (Sec. 8) ---
		se = series_stats(ts_e); smz = series_stats(ts_mz); sQ = series_stats(ts_Q)
		A = w.Sq ./ max(nsq, 1)
		peaks = analyze_peaks(A, Nx, Ny; dom_frac=th.dom_frac, angle_tol=th.angle_tol)
		psi6_vals = filter(!isnan, ts_psi6); psi6l_vals = filter(!isnan, ts_psi6l)
		psi6 = isempty(psi6_vals) ? NaN : mean_of(psi6_vals)
		psi6l = isempty(psi6l_vals) ? NaN : mean_of(psi6l_vals)
		nsk = mean_of(ts_nsk)
		Area = m.N*m.h^2
		R_eff = abs(sQ.mean) ≥ 0.5 ? sqrt(Area/(π*abs(sQ.mean))) : NaN      # Sec. 10.2, N_sk ≈ |Q|
		λ_sites = peaks.q1 > 0 ? 2π/peaks.q1 : NaN
		Qstd = sqrt(var_of(ts_Q))
		label = classify(sQ.mean, sQ.err, Qstd, peaks, psi6, smz.mean, th)
		# --- optional T = 0 quench at the lowest temperature (Sec. 10.1) ---
		ρq = NaN; Qq = NaN
		if is_tmin && cfg.nquench > 0
			Sc = copy(S)
			spq = Sampler(copy(sp.rngs), sp.blocks, sp.order, sp.δ, sp.pfull, sp.E, 0, 0)
			Eq = quench!(Sc, m, spq, cfg.nquench)
			ρq = rho_continuum(Eq/m.N, m)
			Qq, _ = charge_map!(w.qmap, Sc, m)
		end
		keep_full = is_tmin || (tr.ladder == :field && p == length(tr.points))
		Sz = Float32.(S[3, :, :])
		Sqs = Float32.(circshift(A, (Nx ÷ 2, Ny ÷ 2)))
		push!(out, TResult(; hist=tr.hist, seed=tr.seed, b=b, θ=θ, bidx=tr.bidx[p], tidx=tr.tidx[p],
			e=se.mean, e_err=se.err, tau_e=se.tau, neff_e=se.neff,
			e_ex=mean_of(ts_ex), e_dm=mean_of(ts_dm), e_b=mean_of(ts_b), e_k=mean_of(ts_k),
			cV=m.N*var_of(ts_e)/θ^2,
			mz=smz.mean, mz_err=smz.err, tau_mz=smz.tau, chi=m.N*var_of(ts_mz)/θ,
			mperp=mean_of(ts_mperp), sz2=mean_of(ts_sz2),
			Q=sQ.mean, Q_err=sQ.err, Q_std=Qstd, tau_Q=sQ.tau, neff_Q=sQ.neff, absQ=mean_of(abs.(ts_Q)),
			nsk=nsk, psi6=psi6, psi6_local=psi6l, R_eff=R_eff,
			P1=peaks.P1, P2=peaks.P2, P3=peaks.P3, q1x=peaks.q1x, q1y=peaks.q1y, qnorm=peaks.q1,
			λ_sites=λ_sites, λ_lD=λ_sites*m.h, ndom=peaks.ndom, sixfold=peaks.sixfold, square=peaks.square,
			acc=sp.nacc/max(sp.natt, 1), δ=sp.δ, echeck=echeck, degen=ndeg_tot/(2m.N*max(nsq, 1)),
			drift_ok=se.drift_ok && smz.drift_ok && sQ.drift_ok,
			ρ=rho_continuum(se.mean, m), ρ_quench=ρq, Q_quench=Qq,
			label=label, Sz=Sz, Sq=Sqs,
			Sfull=keep_full ? copy(S) : Array{Float64}(undef, 0, 0, 0),
			ts_sweep=ts_sweep, ts_e=ts_e, ts_mz=ts_mz, ts_Qsweep=ts_Qsweep, ts_Q=ts_Q))
	end
	out
end

"""Run all trajectories on a thread pool (one task at a time per worker; inner threads for the checkerboard)."""
function run_all(trajs, cfg, th::Thresholds)
	ntraj = length(trajs)
	nthr = Threads.nthreads()
	# Threads left over after one per trajectory go to the checkerboard sweep, but only when each spawned
	# row block has enough sites (≳ 4096 sites per block) to amortize the task overhead.
	inner = cfg.inner > 0 ? cfg.inner : clamp(cfg.Nx*cfg.Ny ÷ 4096, 1, max(1, nthr ÷ max(ntraj, 1)))
	nworkers = max(1, min(nthr ÷ inner, ntraj))
	results = Vector{Vector{TResult}}(undef, ntraj)
	ch = Channel{Int}(ntraj)
	foreach(k -> put!(ch, k), 1:ntraj); close(ch)
	done = Atomic{Int}(0)
	t0 = time()
	@printf("Running %d trajectories on %d worker(s) × %d inner thread(s) (%d Julia threads)\n",
		ntraj, nworkers, inner, nthr)
	nthr == 1 && println("  hint: start Julia with -t auto so trajectories run in parallel")
	@sync for _ in 1:nworkers
		Threads.@spawn for k in ch
			results[k] = run_trajectory(trajs[k], cfg, inner, th)
			n = atomic_add!(done, 1) + 1
			el = time() - t0
			lock(PRINT_LOCK) do
				tr = trajs[k]
				last = results[k][end]
				@printf("  [%3d/%3d] %-10s seed %d  %s = %.3f  done in %.0f s total (ETA %.0f s)  last: θ=%.3g e=%.4f Q=%.1f %s\n",
					n, ntraj, tr.hist, tr.seed, tr.ladder == :temperature ? "b" : "θ",
					tr.ladder == :temperature ? tr.points[1][1] : tr.points[1][2],
					el, el/n*(ntraj - n), last.θ, last.e, last.Q, last.label)
				flush(stdout)
			end
		end
	end
	results, time() - t0, inner, nworkers
end

# ---------------------------------------------------------------------------
# Combine histories into the (b, θ) diagram (Sec. 9)
# ---------------------------------------------------------------------------
struct Cell
	recs::Vector{TResult}
	final_id::Int
	lowE_id::Int
	agree::Bool
end

"""
Merge the records of all histories/seeds at each grid point. Histories agree when their labels fall in one
phase class. Disagreeing cells are "Unresolved" (resolve = "strict") or take the label of the record with the
lowest mean energy per site (resolve = "energy"); the latter is a low-temperature ground-state proxy
(Sec. 10.1), not a free-energy criterion (Sec. 11.2), and such cells stay marked in the plot and CSV.
"""
function combine_cells(results, nb, nθ, resolve::String)
	cells = Matrix{Cell}(undef, nθ, nb)
	bucket = [TResult[] for _ in 1:nθ, _ in 1:nb]
	for res in results, r in res
		push!(bucket[r.tidx, r.bidx], r)
	end
	for j in 1:nb, k in 1:nθ
		recs = bucket[k, j]
		if isempty(recs)
			cells[k, j] = Cell(recs, UNRESOLVED, UNRESOLVED, false); continue
		end
		ids = [label_id(r.label) for r in recs]
		agree = length(unique(phase_class.(ids))) == 1
		lowE = ids[argmin([r.e for r in recs])]
		final = agree ? ids[1] : (resolve == "energy" ? lowE : UNRESOLVED)
		cells[k, j] = Cell(recs, final, lowE, agree)
	end
	cells
end

# ---------------------------------------------------------------------------
# Output files (Sec. 14)
# ---------------------------------------------------------------------------
fmt(x) = isnan(x) ? "nan" : @sprintf("%.6g", x)

function write_summaries(path, results)
	open(path, "w") do io
		println(io, "history,seed,b,theta,label,e,e_err,tau_e,neff_e,e_exchange,e_dmi,e_field,e_aniso,cV," *
			"mz,mz_err,tau_mz,chi_z,mperp,sz2,Q,Q_err,Q_std,tau_Q,neff_Q,absQ,n_cores,psi6,psi6_local,R_eff," *
			"P1,P2,P3,q1x,q1y,qnorm,lambda_sites,lambda_lD,ndom,sixfold,square,acc,delta,energy_check,degenerate_frac," *
			"drift_ok,rho,rho_quench,Q_quench")
		for res in results, r in res
			vals = (r.e, r.e_err, r.tau_e, r.neff_e, r.e_ex, r.e_dm, r.e_b, r.e_k, r.cV,
				r.mz, r.mz_err, r.tau_mz, r.chi, r.mperp, r.sz2, r.Q, r.Q_err, r.Q_std, r.tau_Q, r.neff_Q, r.absQ,
				r.nsk, r.psi6, r.psi6_local, r.R_eff, r.P1, r.P2, r.P3, r.q1x, r.q1y, r.qnorm, r.λ_sites, r.λ_lD)
			print(io, r.hist, ",", r.seed, ",", @sprintf("%.6f", r.b), ",", @sprintf("%.6g", r.θ), ",", r.label)
			for v in vals
				print(io, ",", fmt(v))
			end
			println(io, ",", r.ndom, ",", r.sixfold, ",", r.square, ",", fmt(r.acc), ",", fmt(r.δ), ",",
				@sprintf("%.3e", r.echeck), ",", fmt(r.degen), ",", r.drift_ok, ",", fmt(r.ρ), ",",
				fmt(r.ρ_quench), ",", fmt(r.Q_quench))
		end
	end
	println("Wrote ", path)
end

function write_phase_csv(path, cells, bs, θs, m0::Model)
	open(path, "w") do io
		println(io, "b,theta,label_id,label,agree,lowest_energy_label,n_records,labels,e_min,rho_helix_continuum")
		for j in eachindex(bs), k in eachindex(θs)
			c = cells[k, j]
			labs = join([LABEL_CHARS[label_id(r.label)] for r in c.recs])
			emin = isempty(c.recs) ? NaN : minimum(r.e for r in c.recs)
			@printf(io, "%.6f,%.6g,%d,%s,%s,%s,%d,%s,%s,%.6f\n", bs[j], θs[k], c.final_id, LABELS[c.final_id],
				c.agree, LABELS[c.lowE_id], length(c.recs), labs, fmt(emin), -0.5 + bs[j] + m0.a/2)
		end
	end
	println("Wrote ", path)
end

function write_timeseries(path, results)
	open(path, "w") do io
		println(io, "history,seed,b,theta,sweep,e,mz,Q")
		for res in results, r in res
			qd = Dict(zip(r.ts_Qsweep, r.ts_Q))
			for (s, e, mz) in zip(r.ts_sweep, r.ts_e, r.ts_mz)
				q = get(qd, s, NaN)
				@printf(io, "%s,%d,%.6f,%.6g,%d,%.8f,%.6f,%s\n", r.hist, r.seed, r.b, r.θ, s, e, mz, fmt(q))
			end
		end
	end
	println("Wrote ", path)
end

function write_snapshots(dir, results)
	mkpath(dir)
	n = 0
	for res in results, r in res
		isempty(r.Sfull) && continue
		name = @sprintf("spins_%s_s%d_b%.3f_theta%.4g.csv", r.hist, r.seed, r.b, r.θ)
		open(joinpath(dir, name), "w") do io
			println(io, "# history=", r.hist, " seed=", r.seed, " b=", r.b, " theta=", r.θ, " label=", r.label,
				" Q=", r.Q, " e=", r.e)
			println(io, "i,j,Sx,Sy,Sz")
			Nx = size(r.Sfull, 2); Ny = size(r.Sfull, 3)
			for j in 1:Ny, i in 1:Nx
				@printf(io, "%d,%d,%.8f,%.8f,%.8f\n", i, j, r.Sfull[1,i,j], r.Sfull[2,i,j], r.Sfull[3,i,j])
			end
		end
		n += 1
	end
	println("Wrote ", n, " spin snapshots to ", dir)
end

function git_revision()
	try
		readchomp(`git -C $(@__DIR__) rev-parse --short HEAD`)
	catch
		"unknown"
	end
end

function write_metadata(path, o, cfg, bs, θs, trajs, elapsed, inner, nworkers, m0::Model)
	open(path, "w") do io
		println(io, "# monte_carlo_2d.jl run metadata (Sec. 14)")
		println(io, "script = \"monte_carlo_2d.jl\"")
		println(io, "git_revision = \"", git_revision(), "\"")
		println(io, "julia = \"", VERSION, "\"")
		println(io, "hostname = \"", gethostname(), "\"")
		println(io, "cpu = \"", Sys.CPU_NAME, "\"")
		println(io, "threads = ", Threads.nthreads())
		println(io, "workers = ", nworkers)
		println(io, "inner_threads = ", inner)
		println(io, "wall_time_s = ", round(elapsed; digits=1))
		println(io, "command_line = \"", join(ARGS, " "), "\"")
		println(io)
		println(io, "[model]")
		println(io, "a = ", cfg.a); println(io, "h = ", cfg.h)
		println(io, "J_lat = ", m0.J); println(io, "D_lat = ", m0.D)
		println(io, "B_lat_per_b = ", m0.J*cfg.h^2, "   # B_lat = b h²"); println(io, "K_lat = ", m0.K)
		println(io, "Nx = ", cfg.Nx); println(io, "Ny = ", cfg.Ny)
		println(io, "Lx = ", cfg.Nx*cfg.h); println(io, "Ly = ", cfg.Ny*cfg.h)
		println(io, "temperature_unit = \"theta = k_B T / J_lat\"")
		println(io, "energy_density_unit = \"rho = (H/N + 2 J_lat + B_lat)/(J_lat h^2), D_c^2/J_c units, relative to saturation\"")
		println(io)
		println(io, "[grid]")
		println(io, "b = [", join(fmt.(bs), ", "), "]")
		println(io, "theta = [", join(fmt.(θs), ", "), "]   # visiting order of cooling runs")
		println(io, "ladder = \"", o["ladder"], "\"")
		println(io)
		println(io, "[sampling]")
		println(io, "update = \"single-spin Metropolis\"")
		println(io, "sweep = \"N = Nx*Ny attempted updates\"")
		println(io, "site_order = \"", o["site_order"], "\"")
		println(io, "proposal = \"cap of half-angle delta (adapted during the first nadapt sweeps of each point, then frozen) with probability 1 - pfull, uniform sphere with probability pfull\"")
		println(io, "pfull = ", cfg.pfull)
		println(io, "rng = \"Xoshiro, seeded from hash((seed, history, seed_index, first_b_index, first_theta_index[, block]))\"")
		println(io, "seed = ", cfg.seed); println(io, "nseeds = ", o["nseeds"])
		println(io, "histories = [", join("\"" .* o["histories"] .* "\"", ", "), "]")
		println(io, "n_trajectories = ", length(trajs))
		println(io, "neq = ", cfg.neq); println(io, "nprod = ", cfg.nprod); println(io, "nadapt = ", cfg.nadapt)
		println(io, "final_factor = ", cfg.final_factor); println(io, "nquench = ", cfg.nquench)
		println(io, "nmeas = ", cfg.nmeas); println(io, "nstruct = ", cfg.nstruct)
		println(io, "skx_seed_cells = [", default_skx_cells(m0)[1], ", ", default_skx_cells(m0)[2], "]")
		println(io, "helix_seed_n = ", best_helix_n(m0))
		println(io, "precision = \"Float64\"")
		println(io)
		println(io, "[classification]")
		for k in ("peak_min", "mz_pol", "qmin", "psi6_min", "dom_frac", "angle_tol", "s_core", "qcore_min")
			println(io, k, " = ", o[k])
		end
		println(io, "resolve = \"", o["resolve"], "\"")
		println(io, "charge_convention = \"Berg-Luscher, triangles (A,B,C),(A,C,D); Bloch core down in up background has Q = -1\"")
	end
	println("Wrote ", path)
end

function print_text_diagram(cells, bs, θs)
	println("\nPhase diagram (rows θ descending, columns b ascending): ",
		join(["$(LABEL_CHARS[i])=$(LABELS[i])" for i in 1:NLABELS], ", "))
	println("  after '|': '.' histories/seeds agree, 'x' they disagree (see --resolve)")
	for k in eachindex(θs)
		@printf("  θ=%-7.4g ", θs[k])
		for j in eachindex(bs)
			print(LABEL_CHARS[cells[k, j].final_id])
		end
		print(" | ")
		for j in eachindex(bs)
			print(cells[k, j].agree ? '.' : 'x')
		end
		println()
	end
	@printf("  %-9s b=%.2f … %.2f\n", "", bs[1], bs[end])
end

# ---------------------------------------------------------------------------
# Plots (Sec. 7.5, 14)
# ---------------------------------------------------------------------------
function cell_edges(v)
	n = length(v)
	n == 1 && return [v[1] - 0.05*max(abs(v[1]), 1.0), v[1] + 0.05*max(abs(v[1]), 1.0)]
	e = similar(v, n + 1)
	for k in 2:n
		e[k] = (v[k-1] + v[k])/2
	end
	e[1] = v[1] - (v[2] - v[1])/2; e[end] = v[end] + (v[end] - v[end-1])/2
	e
end

function save_phase_plot(path, cells, bs, θs, cfg, o)
	ylog = o["ladder"] == "geometric" && length(θs) > 1
	θa = reverse(θs); ya = ylog ? log10.(θa) : θa              # ascending θ
	ye = cell_edges(ya); xe = cell_edges(bs)
	nθ = length(θs); nb = length(bs)
	title = @sprintf("Monte Carlo phases: a = %.2f, h = %.2g, %d × %d, histories %s × %d seed(s)",
		cfg.a, cfg.h, cfg.Nx, cfg.Ny, join(o["histories"], "+"), o["nseeds"])
	p = plot(; xlabel="b = B J_c / D_c²", ylabel="θ = k_B T / J_lat", title=title, titlefontsize=9,
		framestyle=:box, size=(960, 600), dpi=160, legend=:outertopright,
		xlims=(xe[1], xe[end]), ylims=(ye[1], ye[end]))
	for id in 1:NLABELS
		shapes = Shape[]
		for j in 1:nb, ka in 1:nθ
			k = nθ - ka + 1                                   # row in the descending ladder
			cells[k, j].final_id == id || continue
			push!(shapes, Shape([xe[j], xe[j+1], xe[j+1], xe[j]], [ye[ka], ye[ka], ye[ka+1], ye[ka+1]]))
		end
		isempty(shapes) && continue
		plot!(p, shapes; fillcolor=RGB(LABEL_RGB[id]...), linecolor=:white, linewidth=0.4, label="")
		scatter!(p, [NaN], [NaN]; color=RGB(LABEL_RGB[id]...), markershape=:rect, markersize=8,
			markerstrokewidth=0, label=LABELS[id])
	end
	xs = Float64[]; ys = Float64[]; cs = RGB{Float64}[]
	for j in 1:nb, ka in 1:nθ
		k = nθ - ka + 1
		c = cells[k, j]
		(!c.agree && !isempty(c.recs)) || continue
		push!(xs, bs[j]); push!(ys, ya[ka]); push!(cs, RGB(LABEL_RGB[c.lowE_id]...))
	end
	if !isempty(xs)
		if o["resolve"] == "energy"
			scatter!(p, xs, ys; markercolor=:black, markersize=2.2, markerstrokewidth=0,
				label="histories disagree (lowest ⟨E⟩ shown)")
		else
			scatter!(p, xs, ys; markercolor=cs, markersize=4, markerstrokecolor=:black, markerstrokewidth=0.6,
				label="lowest ⟨E⟩ history in unresolved cells")
		end
	end
	if ylog
		ticks = [t for t in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5) if ye[1] ≤ log10(t) ≤ ye[end]]
		plot!(p; yticks=(log10.(ticks), string.(ticks)))
	end
	if cfg.a == 0
		vline!(p, [0.216, 0.801]; linestyle=:dash, color=:black, linewidth=1,
			label="continuum T = 0 (a = 0): b₁ ≈ 0.216, b_c ≈ 0.801")
	end
	savefig(p, path)
	println("Wrote ", path)
end

"""Primary records (first history, seed 1) as a (θ index, b index) matrix of Union{TResult,Nothing}."""
function primary_grid(results, trajs, nb, nθ)
	g = Matrix{Union{TResult,Nothing}}(nothing, nθ, nb)
	for (tr, res) in zip(trajs, results)
		tr.seed == 1 || continue
		for r in res
			g[r.tidx, r.bidx] === nothing && (g[r.tidx, r.bidx] = r)
		end
	end
	g
end

pick_indices(n, k) = n ≤ k ? collect(1:n) : unique(round.(Int, range(1, n; length=k)))

function save_textures(path, g, bs, θs; what=:Sz)
	nθ, nb = size(g)
	js = pick_indices(nb, 6); ks = pick_indices(nθ, 5)
	panels = Any[]
	for k in ks, j in js
		r = g[k, j]
		if r === nothing
			push!(panels, plot(; axis=false, grid=false)); continue
		end
		if what == :Sz
			ttl = @sprintf("b=%.2f θ=%.3g Q=%.0f %s", r.b, r.θ, r.Q, LABEL_CHARS[label_id(r.label)])
			push!(panels, heatmap(1:size(r.Sz, 1), 1:size(r.Sz, 2), permutedims(r.Sz); c=cgrad(:RdBu, rev=true), clims=(-1, 1),
				aspect_ratio=1, colorbar=false, ticks=false, title=ttl, titlefontsize=7, framestyle=:box))
		else
			Nx, Ny = size(r.Sq)
			kx = (1:Nx) .- (Nx ÷ 2 + 1); ky = (1:Ny) .- (Ny ÷ 2 + 1)
			ttl = @sprintf("b=%.2f θ=%.3g P1=%.3f ndom=%d", r.b, r.θ, r.P1, r.ndom)
			push!(panels, heatmap(kx, ky, permutedims(log10.(r.Sq .+ 1f-9)); c=:viridis, clims=(-6, 0),
				aspect_ratio=1, colorbar=false, ticks=false, title=ttl, titlefontsize=7, framestyle=:box))
		end
	end
	p = plot(panels...; layout=(length(ks), length(js)), size=(230*length(js), 235*length(ks)), dpi=140,
		plot_title=what == :Sz ? "S_z snapshots (x horizontal, y vertical), primary history" :
		                         "log10 S(q)/N averaged over snapshots, q = 0 at centre", plot_titlefontsize=10)
	savefig(p, path)
	println("Wrote ", path)
end

function save_observables(path, g, bs, θs, o)
	nθ, nb = size(g)
	js = pick_indices(nb, 6)
	xsc = o["ladder"] == "geometric" && nθ > 1 ? :log10 : :identity
	panels = Any[]
	for (name, f) in (("e / site", r -> r.e), ("c_V", r -> r.cV), ("⟨m_z⟩", r -> r.mz),
	                  ("χ_z", r -> r.chi), ("|⟨Q⟩|", r -> abs(r.Q)), ("max S(q)/N", r -> r.P1))
		p = plot(; xlabel="θ", ylabel=name, xscale=xsc, legend=(name == "e / site" ? :topright : false),
			framestyle=:box, titlefontsize=9)
		for j in js
			xs = Float64[]; ys = Float64[]
			for k in 1:nθ
				r = g[k, j]; r === nothing && continue
				push!(xs, r.θ); push!(ys, f(r))
			end
			isempty(xs) && continue
			plot!(p, xs, ys; marker=:circle, markersize=2.5, label=@sprintf("b = %.2f", bs[j]))
		end
		push!(panels, p)
	end
	p = plot(panels...; layout=(2, 3), size=(1200, 700), dpi=140, plot_title="Primary history vs temperature",
		plot_titlefontsize=10)
	savefig(p, path)
	println("Wrote ", path)
end

function save_convergence(path, g, bs, θs)
	nθ, nb = size(g)
	j = argmin(abs.(bs .- 0.6))
	ks = pick_indices(nθ, 3)
	panels = Any[]
	for k in ks
		r = g[k, j]
		if r === nothing
			for _ in 1:3; push!(panels, plot(; axis=false)); end
			continue
		end
		ttl = @sprintf("b=%.2f θ=%.3g τ_e=%.1f τ_mz=%.1f τ_Q=%.1f", r.b, r.θ, r.tau_e, r.tau_mz, r.tau_Q)
		push!(panels, plot(r.ts_sweep, r.ts_e; xlabel="sweep", ylabel="e / site", title=ttl, titlefontsize=7, legend=false))
		push!(panels, plot(r.ts_sweep, r.ts_mz; xlabel="sweep", ylabel="m_z", legend=false))
		qlo, qhi = extrema(r.ts_Q)
		qhi - qlo < 1 && (qlo -= 1; qhi += 1)
		push!(panels, plot(r.ts_Qsweep, r.ts_Q; xlabel="sweep", ylabel="Q", marker=:circle, markersize=2, legend=false,
			ylims=(qlo, qhi)))
	end
	p = plot(panels...; layout=(length(ks), 3), size=(1200, 300*length(ks)), dpi=140,
		plot_title="Production traces, primary history (τ in measurement intervals)", plot_titlefontsize=10)
	savefig(p, path)
	println("Wrote ", path)
end

# ---------------------------------------------------------------------------
# Self-test (Sec. 13.2) and benchmark
# ---------------------------------------------------------------------------
function selftest(o)
	checks = Tuple{String,Bool,String}[]
	rng = Xoshiro(2024)
	L = 16
	m = Model(L, L; h=0.5, b=0.6, a=0.2)
	S = random_spins!(Array{Float64}(undef, 3, L, L), rng)

	# 1. unit length after sampling
	sp = Sampler([Xoshiro(1)], [1:L], :checkerboard, 1.0, 0.05, total_energy(S, m), 0, 0)
	for _ in 1:50; sweep!(S, m, sp, 1/0.5); end
	dev = maximum(abs(sqrt(S[1,i,j]^2 + S[2,i,j]^2 + S[3,i,j]^2) - 1) for j in 1:L, i in 1:L)
	push!(checks, ("Unit length after 50 sweeps", dev < 1e-12, @sprintf("max |‖S‖−1| = %.1e", dev)))

	# 2. local ΔH vs full difference (random sites, including boundary rows/columns)
	worst = 0.0
	for t in 1:200
		i = t ≤ 50 ? rand(rng, (1, L)) : rand(rng, 1:L); j = t ≤ 50 ? rand(rng, 1:L) : rand(rng, (1, L))
		E0 = total_energy(S, m)
		sx, sy, sz = S[1,i,j], S[2,i,j], S[3,i,j]
		hx, hy, hz = local_field(S, m, i, j)
		nx, ny, nz = uniform_sphere(rng)
		ΔH = -((nx - sx)*hx + (ny - sy)*hy + (nz - sz)*hz) - m.K*(nz*nz - sz*sz)
		S[1,i,j], S[2,i,j], S[3,i,j] = nx, ny, nz
		worst = max(worst, abs(ΔH - (total_energy(S, m) - E0)))
	end
	push!(checks, ("Local ΔH matches full energy difference (200 moves)", worst < 1e-10, @sprintf("max discrepancy %.1e", worst)))

	# 3. uniform +z energy
	fill!(S, 0.0); S[3, :, :] .= 1.0
	Eex, Edm, Eb, Ek = energy_components(S, m)
	ok3 = abs(Eex/m.N + 2m.J) < 1e-13 && abs(Edm) < 1e-13 && abs(Eb/m.N + m.B) < 1e-13 && abs(Ek) < 1e-13
	push!(checks, ("Saturated state: e = −2J − B, 2N forward bonds", ok3, @sprintf("e = %.6f, expected %.6f", (Eex+Edm+Eb+Ek)/m.N, -2m.J - m.B)))

	# 4. helix energies and handedness (b = a = 0)
	m0 = Model(L, L; h=0.5)
	n = 2; k = 2π*n/L
	diagonal_helix!(S, m0, n)
	e_diag = total_energy(S, m0)/m0.N; e_diag_exp = -2m0.J*cos(k) - √2*m0.D*sin(k)
	helix_seed!(S, m0; n=n)
	e_ax = total_energy(S, m0)/m0.N; e_ax_exp = axial_helix_energy(m0, k)
	S[2, :, :] .*= -1                               # wrong helicity
	e_wrong = total_energy(S, m0)/m0.N
	ok4 = abs(e_diag - e_diag_exp) < 1e-12 && abs(e_ax - e_ax_exp) < 1e-12 && e_wrong > e_ax
	push!(checks, ("Diagonal/axial helix energies analytic; D > 0 helicity lower", ok4,
		@sprintf("diag %.6f (exp %.6f), axial %.6f (exp %.6f), wrong helicity %.6f", e_diag, e_diag_exp, e_ax, e_ax_exp, e_wrong)))
	# optimum of the lattice helix approaches |q| = 1 (atan(h)/h → 1)
	kopt = atan(m0.D/m0.J); qphys = kopt/m0.h
	push!(checks, ("Axial helix optimum |q| ℓ_D = atan(h)/h → 1 as h → 0", abs(qphys - 1) < m0.h^2/3 + 1e-12,
		@sprintf("|q| ℓ_D = %.4f at h = %.2f", qphys, m0.h)))

	# 5. topological charge convention
	L2 = 48; m2 = Model(L2, L2; h=0.25)
	S2 = Array{Float64}(undef, 3, L2, L2); q2 = zeros(L2, L2)
	bloch_seed!(S2, m2; R0=12.0)
	Qb, _ = charge_map!(q2, S2, m2)
	fill!(S2, 0.0); S2[3, :, :] .= 1.0
	Qu, _ = charge_map!(q2, S2, m2)
	skx_seed!(S2, m2)
	nx, my = default_skx_cells(m2)
	Qs, _ = charge_map!(q2, S2, m2)
	push!(checks, ("Charge: Bloch seed Q = −1, uniform Q = 0, triple-q seed Q = −nx·my",
		abs(Qb + 1) < 1e-8 && abs(Qu) < 1e-12 && abs(Qs + nx*my) < 1e-8,
		@sprintf("Q_bloch = %.6f, Q_uniform = %.1e, Q_skx = %.3f (nx·my = %d)", Qb, Qu, Qs, nx*my)))
	# core detector on the seed (needs the charge map in the work space)
	w2 = Work(L2, L2)
	charge_map!(w2.qmap, S2, m2)
	ncore = detect_cores!(w2, S2, m2)
	g6, _ = psi6_order(w2.centres, L2, L2)
	push!(checks, ("Core detector finds nx·my cores with |ψ6| > 0.9 on the seed", ncore == nx*my && g6 > 0.9,
		@sprintf("%d cores, |ψ6| = %.3f", ncore, g6)))
	# structure factor: triple-q seed is sixfold, helix is single-pair
	fill!(w2.Sq, 0.0); fill!(w2.Szz, 0.0)
	accumulate_sf!(w2.Sq, w2.Szz, S2, m2, w2.sf)
	pk = analyze_peaks(w2.Sq, L2, L2)
	helix_seed!(S2, m2; n=2)
	fill!(w2.Sq, 0.0); fill!(w2.Szz, 0.0)
	accumulate_sf!(w2.Sq, w2.Szz, S2, m2, w2.sf)
	ph = analyze_peaks(w2.Sq, L2, L2)
	push!(checks, ("Structure factor: seed sixfold (3 dominant pairs), helix single pair at q = 2πn/L",
		pk.sixfold && pk.ndom == 3 && ph.ndom == 1 && abs(ph.q1 - 2π*2/L2) < 1e-12 && abs(ph.P1 - 0.5) < 1e-12,
		@sprintf("seed ndom = %d sixfold = %s; helix ndom = %d |q1| = %.4f P1 = %.4f", pk.ndom, pk.sixfold, ph.ndom, ph.q1, ph.P1)))

	# 6a. thermal sampler: pure exchange at high temperature, e ≈ −2J²/(3θ)
	mJ = Model(L, L; h=0.5, D=0.0, B=0.0, K=0.0)
	random_spins!(S, rng)
	θ = 50.0
	sp = Sampler([Xoshiro(7)], [1:L], :checkerboard, 2.0, 0.05, total_energy(S, mJ), 0, 0)
	for _ in 1:200; sweep!(S, mJ, sp, 1/θ); end
	es = Float64[]
	for s in 1:4000
		sweep!(S, mJ, sp, 1/θ)
		s % 4 == 0 && push!(es, total_energy(S, mJ)/mJ.N)
	end
	st = series_stats(es); e_exp = -2mJ.J^2/(3θ)
	push!(checks, ("Pure exchange at θ = 50: e ≈ −2J²/(3θ) within 3σ", abs(st.mean - e_exp) < 3st.err + 1e-3,
		@sprintf("e = %.5f ± %.5f, expected %.5f", st.mean, st.err, e_exp)))
	# 6b. independent spins in a field: m_z = coth x − 1/x
	mB = Model(L, L; h=0.5, J=0.0, D=0.0, B=1.0, K=0.0)
	θ = 0.5; x = mB.B/θ
	sp = Sampler([Xoshiro(8)], [1:L], :checkerboard, 1.0, 0.05, total_energy(S, mB), 0, 0)
	for _ in 1:200; sweep!(S, mB, sp, 1/θ); end
	ms = Float64[]
	for s in 1:4000
		sweep!(S, mB, sp, 1/θ)
		s % 4 == 0 && push!(ms, magnetization(S, mB)[3])
	end
	st = series_stats(ms); m_exp = coth(x) - 1/x
	push!(checks, ("Independent spins, B/θ = 2: ⟨m_z⟩ = coth x − 1/x within 3σ", abs(st.mean - m_exp) < 3st.err + 1e-3,
		@sprintf("m_z = %.5f ± %.5f, expected %.5f", st.mean, st.err, m_exp)))
	# 6c. same with the random-site order
	sp = Sampler([Xoshiro(9)], [1:L], :random, 1.0, 0.05, total_energy(S, mB), 0, 0)
	for _ in 1:200; sweep!(S, mB, sp, 1/θ); end
	ms = Float64[]
	for s in 1:4000
		sweep!(S, mB, sp, 1/θ)
		s % 4 == 0 && push!(ms, magnetization(S, mB)[3])
	end
	st = series_stats(ms)
	push!(checks, ("Random-site order gives the same ⟨m_z⟩ within 3σ", abs(st.mean - m_exp) < 3st.err + 1e-3,
		@sprintf("m_z = %.5f ± %.5f", st.mean, st.err)))

	# 7. cached energy, integer Q and repeatability
	random_spins!(S, rng)
	run_once(seed) = begin
		Sr = random_spins!(Array{Float64}(undef, 3, L, L), Xoshiro(seed))
		spr = Sampler([Xoshiro(seed + 1)], [1:L], :checkerboard, 1.0, 0.05, total_energy(Sr, m), 0, 0)
		for _ in 1:300; sweep!(Sr, m, spr, 1/0.3); end
		Sr, spr
	end
	Sa, spa = run_once(11); Sb, spb = run_once(11); Sc, spc = run_once(12)
	ecache = abs(spa.E - total_energy(Sa, m))
	push!(checks, ("Cached energy equals full recomputation after 300 sweeps", ecache < 1e-8, @sprintf("|ΔE| = %.1e", ecache)))
	push!(checks, ("Same seed reproduces the configuration exactly", Sa == Sb, ""))
	Qa, _ = charge_map!(zeros(L, L), Sa, m)
	push!(checks, ("Berg–Lüscher charge of a thermal configuration is an integer", abs(Qa - round(Qa)) < 1e-8, @sprintf("Q = %.8f", Qa)))
	# 7b. parallel checkerboard (blocks) keeps the energy cache consistent and is valid at the uncoupled-spin test
	if Threads.nthreads() > 1 || true
		Sp = random_spins!(Array{Float64}(undef, 3, L, L), Xoshiro(21))
		spp = Sampler([Xoshiro(k) for k in 1:4], row_blocks(L, 4), :checkerboard, 1.0, 0.05, total_energy(Sp, mB), 0, 0)
		for _ in 1:200; sweep!(Sp, mB, spp, 1/0.5); end
		ms = Float64[]
		for s in 1:4000
			sweep!(Sp, mB, spp, 1/0.5)
			s % 4 == 0 && push!(ms, magnetization(Sp, mB)[3])
		end
		st = series_stats(ms)
		ecache = abs(spp.E - total_energy(Sp, mB))
		push!(checks, ("4-block checkerboard sweep: energy cache exact, ⟨m_z⟩ within 3σ",
			ecache < 1e-8 && abs(st.mean - m_exp) < 3st.err + 1e-3, @sprintf("|ΔE| = %.1e, m_z = %.5f ± %.5f", ecache, st.mean, st.err)))
		spp = Sampler([Xoshiro(k) for k in 1:4], row_blocks(L, 4), :checkerboard, 1.0, 0.05, total_energy(Sp, m), 0, 0)
		for _ in 1:100; sweep!(Sp, m, spp, 1/0.3); end
		ecache = abs(spp.E - total_energy(Sp, m))
		push!(checks, ("4-block checkerboard sweep with DMI and anisotropy: energy cache exact", ecache < 1e-8, @sprintf("|ΔE| = %.1e", ecache)))
	end
	# 8. quench of the triple-q seed stays a skyrmion lattice and lowers the energy
	skx_seed!(S2, m2)
	m2b = with_b(m2, 0.6)
	spq = Sampler([Xoshiro(5)], [1:L2], :checkerboard, 0.3, 0.05, total_energy(S2, m2b), 0, 0)
	E0 = spq.E
	Eq = quench!(S2, m2b, spq, 300)
	Qq, _ = charge_map!(q2, S2, m2b)
	push!(checks, ("T = 0 quench of the seed at b = 0.6 lowers E and keeps Q = −nx·my", Eq < E0 && abs(Qq + nx*my) < 1e-8,
		@sprintf("ρ: %.4f → %.4f, Q = %.3f", rho_continuum(E0/m2b.N, m2b), rho_continuum(Eq/m2b.N, m2b), Qq)))

	println("\nSelf-test (README_monte_carlo_2d.md Sec. 13.2)")
	println("-"^100)
	nfail = 0
	for (name, ok, info) in checks
		ok || (nfail += 1)
		@printf("  %-4s %-78s %s\n", ok ? "PASS" : "FAIL", name, info)
	end
	println("-"^100)
	println(length(checks), " checks, ", nfail, " failed")
	return nfail
end

function bench(o)
	Nx, Ny = o["Nx"], o["Ny"]
	m = Model(Nx, Ny; h=o["h"], b=0.6, a=o["a"])
	S = random_spins!(Array{Float64}(undef, 3, Nx, Ny), Xoshiro(1))
	for (name, order, nbl) in (("checkerboard, 1 block", :checkerboard, 1),
	                           ("random site order", :random, 1),
	                           (@sprintf("checkerboard, %d blocks (threads)", Threads.nthreads()), :checkerboard, Threads.nthreads()))
		sp = Sampler([Xoshiro(k) for k in 1:nbl], row_blocks(Ny, nbl), order, 0.8, o["pfull"], total_energy(S, m), 0, 0)
		for _ in 1:20; sweep!(S, m, sp, 1/0.3); end        # compile + warm up
		nsw = 200
		t = @elapsed for _ in 1:nsw; sweep!(S, m, sp, 1/0.3); end
		@printf("  %-40s %7.2f M attempts/s  (%.2f ms per sweep of %d sites, acc %.2f)\n",
			name, nsw*m.N/t/1e6, t/nsw*1e3, m.N, sp.nacc/sp.natt)
	end
	w = Work(Nx, Ny)
	t = @elapsed for _ in 1:10; accumulate_sf!(w.Sq, w.Szz, S, m, w.sf); end
	@printf("  structure factor (3 components)      %7.2f ms per snapshot\n", t/10*1e3)
	t = @elapsed for _ in 1:10; charge_map!(w.qmap, S, m); detect_cores!(w, S, m); end
	@printf("  charge map + core detection          %7.2f ms per snapshot\n", t/10*1e3)
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main(o)
	if o["selftest"]
		exit(selftest(o) == 0 ? 0 : 1)
	end
	if o["bench"]
		bench(o); exit(0)
	end
	bs = o["b"] !== nothing ? [o["b"]] : axis_grid(o["bmin"], o["bmax"]; n=o["nb"], step=o["db"])
	θs = o["theta"] !== nothing ? [o["theta"]] : temperature_ladder(o["theta_max"], o["theta_min"], o["ntheta"], o["ladder"])
	cfg = (; Nx=o["Nx"], Ny=o["Ny"], h=o["h"], a=o["a"], seed=o["seed"], site_order=o["site_order"], inner=o["inner"],
		pfull=o["pfull"], neq=o["neq"], nprod=o["nprod"], nadapt=o["nadapt"], final_factor=o["final_factor"],
		nquench=o["nquench"], nmeas=o["nmeas"], nstruct=o["nstruct"], helix_n=o["helix_n"], skx_nx=o["skx_nx"], skx_my=o["skx_my"])
	th = Thresholds(o["peak_min"], o["mz_pol"], o["qmin"], o["psi6_min"], o["dom_frac"], deg2rad(o["angle_tol"]),
		o["s_core"], o["qcore_min"])
	trajs = build_trajectories(o["histories"], o["nseeds"], bs, θs)
	m0 = Model(cfg.Nx, cfg.Ny; h=cfg.h, b=bs[1], a=cfg.a)
	run_id = isempty(o["run_id"]) ?
		@sprintf("a%s_h%s_L%dx%d_%s", replace(string(cfg.a), "." => "p"), replace(string(cfg.h), "." => "p"),
			cfg.Nx, cfg.Ny, Dates.format(now(), "yyyymmdd-HHMMSS")) : o["run_id"]
	outdir = joinpath(o["outdir"], run_id)
	mkpath(outdir); mkpath(joinpath(outdir, "plots"))

	println("monte_carlo_2d.jl — Metropolis Monte Carlo of the 2D chiral magnet (README_monte_carlo_2d.md)")
	println("Started ", now())
	@printf("Model: a = %.3f, h = %.3g, lattice %d × %d (L = %.1f × %.1f ℓ_D), J = 1, D = %.3g, K = %.3g, B = b·%.3g\n",
		cfg.a, cfg.h, cfg.Nx, cfg.Ny, cfg.Nx*cfg.h, cfg.Ny*cfg.h, m0.D, m0.K, m0.J*cfg.h^2)
	@printf("Lattice helix wavelength: %.2f sites (axial seed n = %d → %.2f sites); triple-q seed cells %d × %d = %d skyrmions\n",
		2π/atan(m0.D/m0.J), best_helix_n(m0), cfg.Nx/best_helix_n(m0), default_skx_cells(m0)..., prod(default_skx_cells(m0)))
	@printf("Grid: b = %s … %s (%d), θ = %s … %s (%d, %s ladder); histories %s × %d seed(s) → %d trajectories, %d grid points\n",
		fmt(bs[1]), fmt(bs[end]), length(bs), fmt(θs[1]), fmt(θs[end]), length(θs), o["ladder"],
		join(o["histories"], ","), o["nseeds"], length(trajs), length(bs)*length(θs))
	@printf("Budget per point: %d equilibration (%d adaptation) + %d production sweeps; measurements every %d, structure every %d; final factor %d; quench %d\n",
		cfg.neq, cfg.nadapt, cfg.nprod, cfg.nmeas, cfg.nstruct, cfg.final_factor, cfg.nquench)
	@printf("Total attempted updates ≈ %.3g\n",
		sum(length(tr.points) for tr in trajs)*(cfg.neq + cfg.nprod)*cfg.Nx*cfg.Ny*1.0)
	println("Output: ", outdir)

	results, elapsed, inner, nworkers = run_all(trajs, cfg, th)
	@printf("All trajectories finished in %.1f s (%.2f M attempts/s aggregate)\n", elapsed,
		sum(length(tr.points) for tr in trajs)*(cfg.neq + cfg.nprod)*cfg.Nx*cfg.Ny/elapsed/1e6)

	cells = combine_cells(results, length(bs), length(θs), o["resolve"])
	print_text_diagram(cells, bs, θs)
	println("\nLabel counts on the grid:")
	for id in 1:NLABELS
		n = count(c -> c.final_id == id, cells)
		n > 0 && @printf("  %-24s %4d\n", LABELS[id], n)
	end
	nd = count(c -> !isempty(c.recs) && any(r -> !r.drift_ok, c.recs), cells)
	@printf("Cells with a drifting production trace in some history: %d of %d (see drift_ok in summaries.csv)\n",
		nd, length(cells))
	if length(θs) ≥ 1
		println("\nLowest temperature θ = ", fmt(θs[end]), ": continuum-scaled energy densities ρ (D_c²/J_c, relative to saturation)")
		@printf("  %-6s %-10s %-6s %-10s %-10s %-10s %-8s %s\n", "b", "history", "seed", "⟨ρ⟩", "ρ_quench", "ρ_helix", "Q", "label")
		for res in results, r in res
			r.tidx == length(θs) || continue
			@printf("  %-6.3f %-10s %-6d %-10.4f %-10.4f %-10.4f %-8.1f %s\n", r.b, r.hist, r.seed, r.ρ, r.ρ_quench,
				-0.5 + r.b + cfg.a/2, r.Q, r.label)
		end
	end

	write_summaries(joinpath(outdir, "summaries.csv"), results)
	write_phase_csv(joinpath(outdir, "phase_diagram.csv"), cells, bs, θs, m0)
	o["timeseries"] && write_timeseries(joinpath(outdir, "observables.csv"), results)
	o["snapshots"] && write_snapshots(joinpath(outdir, "snapshots"), results)
	write_metadata(joinpath(outdir, "metadata.toml"), o, cfg, bs, θs, trajs, elapsed, inner, nworkers, m0)
	if o["plots"] && NEED_PLOTS
		g = primary_grid(results, trajs, length(bs), length(θs))
		save_phase_plot(joinpath(outdir, "plots", "phase_diagram.png"), cells, bs, θs, cfg, o)
		save_textures(joinpath(outdir, "plots", "spin_textures.png"), g, bs, θs; what=:Sz)
		save_textures(joinpath(outdir, "plots", "structure_factors.png"), g, bs, θs; what=:Sq)
		save_observables(joinpath(outdir, "plots", "observables.png"), g, bs, θs, o)
		save_convergence(joinpath(outdir, "plots", "convergence.png"), g, bs, θs)
	end
	println("Done ", now())
end

main(parse_args(ARGS))
