module GaussianPropagator

using LinearAlgebra
using DCATools
using DCAMCMC
using BenchmarkTools
using Optim

import Base:exp, ^, +, *
export *, StatPropagator, PairwiseStat

function *(F::LinearAlgebra.Eigen, x::Array{Float64,1})
	y = zeros(Float64,length(x))
	for i in 1:length(x)
		α = 0.
		for a in 1:length(y)
			α += F.vectors[a,i] * x[a]
		end
		α *= F.values[i]
		for a in 1:length(y)
			y[a] += α * F.vectors[a,i]
		end
	end
	return y
end

struct StatPropagator
	m::Array{Float64,1} # Mean at equilibrium
	C::LinearAlgebra.Eigen # Eig decomposition of the equilibrium C
	x0::Array{Float64,1} # Origin of diffusion
	γ::Float64 # Inverse of characteristic time
	L::Int64
	q::Int64
end


struct PairwiseStat
	m::Array{Float64,1}
	C::Array{Float64,2}
end


include("propagation.jl")
include("misc.jl")


end
