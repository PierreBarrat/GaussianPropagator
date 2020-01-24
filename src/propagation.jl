function propagate(P::StatPropagator, t::Float64)
	pC = propagate(P.C,  P.γ, t)
	pm = propagate(P.m, P.x0, P.C, P.γ, t)
	return pm, pC
end

"""
	propagate(C::LinearAlgebra.Eigen, γ::Float64, t::Float64)

Return the `Eigen` object corresponding to the matrix `C*(1-Λ^2t)`.
"""
function propagate(C::LinearAlgebra.Eigen, γ::Float64, t::Float64)
	peigvals = map(x->x*(1-exp(-2*γ*t/x)), C.values)
	return Eigen(peigvals, C.vectors)
end

"""
	propagate(m::Array{Float64,1}, x0::Array{Float64,1}, C::LinearAlgebra.Eigen, γ::Float64, t::Float64)
"""
function propagate(m::Array{Float64,1}, x0::Array{Float64,1}, C::LinearAlgebra.Eigen, γ::Float64, t::Float64)
	peigvals = map(x->exp(-γ*t/x), C.values)
	return m + Eigen(peigvals, C.vectors)*(x0 - m)
end

function get_Lambda(P,t)
	leigvals = map(x->exp(-2*γ*t/x), P.C.values)
	return Eigen(leigvals, P.C.vectors)
end