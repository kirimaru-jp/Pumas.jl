using Test
using Pumas

# Gut dosing model
m_diffeq = @model begin
    @param begin
        θ ∈ VectorDomain(3, lower=zeros(3), init=ones(3))
    end

    @random begin
        η ~ MvNormal(Matrix{Float64}(I, 2, 2))
    end

    @pre begin
        Ka = θ[1]
        CL = θ[2]*exp(η[1])
        V  = θ[3]*exp(η[2])
    end

    @vars begin
        cp = Central/V
    end

    @dynamics begin
        Depot'   = -Ka*Depot
        Central' =  Ka*Depot - CL*cp
    end

    @derived begin
        conc = @. Central / V
        dv ~ @. Normal(conc, 1e-100)
    end
end

###############################
# Test 15
###############################


param = (θ = [
     1.5,  #Ka
     1.0,  #CL
     30.0  #V
     ],)
randeffs = init_randeffs(m_diffeq, param)

subject = Subject(evs = DosageRegimen([10, 20], ii = 24, addl = 2, ss = 1:2, time = [0, 12], cmt = 2))
sol = solve(m_diffeq, subject, param, randeffs; tspan=(0.0,12.0+1e-14), abstol=1e-14, reltol=1e-14)
col = pre(m_diffeq, subject, param, randeffs)
@test [1000 * sol(12*i)[2] / col(0).V for i in 0:1] ≈ [605.3220736386598;1616.4036675452326] atol=1e-8
###############################
# Test 16
###############################

subject = Subject(evs = DosageRegimen([10, 20, 10], ii = 24, ss = [1,2,1], time = 0:12:24, cmt = 2))
col = pre(m_diffeq, subject, param, randeffs)
sol = solve(m_diffeq, subject, param, randeffs; saveat = 0.0:12.0:60.0, abstol=1e-14,reltol=1e-14)

@test [1000 * sol(12*i)[2] / col(0).V for i in 0:5] ≈ [605.3220736386598
                                                        1616.4036675452326
                                                        605.3220736387212
                                                        405.75952026789673
                                                        271.98874030537564
                                                        182.31950492267478] atol=1e-9
