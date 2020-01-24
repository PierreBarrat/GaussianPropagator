using GaussianPropagator
using LinearAlgebra
using Distributions

function test_simple_GP()
	λ = 2.
	C = [1 0 ; 0 λ]
	m = [0., 0.]
	x0 = [1., 1.]
	γ = 1.

	out = []
	GP = StatPropagator(m, eigen(C), x0, γ, 2, 1)
	for t in 0.1:0.1:10.
		pm_th = [exp(-γ*t), exp(-γ*t/λ)]
		pC_th = [1-exp(-2*γ*t) 0 ; 0 (λ - λ*exp(-2*γ*t/λ))]
		pm, pC = GaussianPropagator.propagate(GP, t)
		pC = GaussianPropagator.mat_from_eig(pC)
		push!(out, (t, pm_th, pm, pC_th, pC))
		println("$(pm_th - pm)")
	end
	return out
end

function test_fit_gamma()
	λ = 2.
	C = [1 0 ; 0 λ]
	m = [0., 0.]
	γ = 1.
	F = eigen(C)

	Peq = MvNormal(m,C)

	Z = Array{Float64,1}(undef,0)
	ts = Array{Float64,1}(undef,0)
	for i in 1:10000
		# Propagator for some init
		x0 = vec(rand(Peq, 1))
		GP = StatPropagator(m, F, x0, γ, 2, 1)
		# Propagating this for time t
		t = abs(randn(1)[1]) + 0.01
		# t = 1.
		pm, pC = GaussianPropagator.propagate(GP, t)
		pC = GaussianPropagator.mat_from_eig(pC)
		# println(pC)
		# Sampling from the propagated model
		P = MvNormal(pm,pC)
		X = rand(P, 2)
		# println(X)
		push!(Z, dot(X[:,1] - m, x0 - m))
		push!(ts, t)
	end
	
	
	return Z,ts, F


end

function test1(a::Array{Float64,1},b::Array{Float64,1})
	# return sum(a.*b)
	# return mapreduce(x->x[1]+x[2], +, zip(a,b))
	# return dot(a,b)
	out = 0.
	for i in 1:length(b)
		out += a[i]*b[i]
	end
	return out
end
function timetest()
	C = randn(500,500)
	C += C'
	F = eigen(C)
	x = randn(500)
	@time F*x
	return nothing
end

