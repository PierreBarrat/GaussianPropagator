function mat_from_eig(F::LinearAlgebra.Eigen)
	return F.vectors*Diagonal(F.values)*F.vectors'
end

"""
	fit_gamma(Z,t,GP::StatPropagator)

Find the optimal timescale `γ`. `Z` contains scalar products of samples, *i.e.* of the form `x1 . x2`, and `t` the times separating the two corresponding samples `x1` and `x2`. 
"""
function fit_gamma(Z::Array{Float64,1},t,C::Eigen)
	Zc::Array{Float64,1} = copy(Z)
	tc::Array{Float64,1} = Float64.(copy(t))
	ρ::Array{Float64,1} = copy(C.values)

	f = let Z=Zc, t=tc, ρ=ρ
		γ -> begin
		    out = 0.
		    for i in 1:length(Z)
		    	tmp = Z[i]
		    	for (a,r) in enumerate(ρ)
		    		tmp -= r*exp(-γ*t[i]/r)
		    	end
		    	out += tmp^2
		    end
		    out
		end
	end
	opt = optimize(x->f(first(x)), [1.], BFGS())
	return opt.minimizer, opt, f
end

"""
	felpropagate(s0,ω,𝑈,t;q=21)

Felsenstein's model of evolution
"""
function felpropagate(s0,ω,𝑈,t;q=21)
	f1 = exp(-𝑈*t)*s0 .+ (1-exp(-𝑈*t))*ω[1]
	L = Int64(length(s0)/q)
	f2 = zeros(L*q, L*q)
	for i in 1:L
		ri = (i-1)*q .+ (1:q)
		s0i = findfirst(x->x==1, s0[ri])
		for j in (i+1):L
			rj = (j-1)*q .+ (1:q)
			s0j = findfirst(x->x==1, s0[rj])
			# println(typeof(ω[2][(i-1)*q + findfirst(x->x==1, s0[ri]), rj]'))
			f2[ri, rj] .+= s0[ri]*ω[2][(i-1)*q + s0i, rj]'/ω[1][(i-1)*q + s0i]
			f2[ri, rj] .+= (s0[rj]*ω[2][(j-1)*q + s0j, ri]'/ω[1][(j-1)*q + s0j])'
		end
	end
	f2 .= (1-exp(-𝑈*t))*exp(-𝑈*t)*(f2 + f2')
	f2 .+= exp(-2*𝑈*t)*s0*s0' .+ (1-exp(-𝑈*t))^2*ω[2]
	return f1, f2
end