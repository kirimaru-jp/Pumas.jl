using Pumas, Test, Random, LabelledArrays

# Read the data# Read the data
data = read_pumas(example_data("data1"),
                      cvs = [:sex,:wt,:etn])
# Cut off the `t=0` pre-dose observation as it throws conditional_nll calculations
# off the scale (variance of the simulated distribution is too small).
for subject in data
    if subject.time[1] == 0
        popfirst!(subject.time)
        popfirst!(subject.observations.dv)
    end
end

## parameters
mdsl = @model begin
    @param begin
        θ ∈ VectorDomain(4, lower=zeros(4), init=ones(4))
        Ω ∈ PSDDomain(2)
        Σ ∈ RealDomain(lower=0.0, init=1.0)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @covariates sex wt etn

    @pre begin
        Ka = θ[1]
        CL = θ[2] * ((wt/70)^0.75) * (θ[4]^sex) * exp(η[1])
        V  = θ[3] * exp(η[2])
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
        dv ~ @. Normal(conc, conc*Σ)
    end
end

p = ParamSet((θ = VectorDomain(4, lower=zeros(4), init=ones(4)), # parameters
              Ω = PSDDomain(2),
              Σ = RealDomain(lower=0.0, init=1.0),
              a = ConstDomain(0.2)))

function rfx_f(p)
    ParamSet((η=MvNormal(p.Ω),))
end

function col_f(param,randeffs,subject)
  function pre(t)
      cov = subject.covariates
      (Σ  = param.Σ,
      Ka = param.θ[1],  # pre
      CL = param.θ[2] * ((cov.wt/70)^0.75) *
           (param.θ[4]^cov.sex) * exp(randeffs.η[1]),
      V  = param.θ[3] * exp(randeffs.η[2]))
    end
end

OneCompartmentVector = @SLVector (:Depot,:Central)
function init_f(col,t0)
     OneCompartmentVector(0.0,0.0)
end

function static_onecompartment_f(u,p,t)
    OneCompartmentVector(-p.Ka*u[1], p.Ka*u[1] - (p.CL/p.V)*u[2])
end
prob = ODEProblem(static_onecompartment_f,nothing,nothing,nothing)

function derived_f(col,sol,obstimes,subject,param,random)
    _col = col(0.0)
    central = sol(obstimes;idxs=2)
    conc = @. central / _col.V
    dv = @. Normal(conc, conc*param.Σ)
    (dv = dv,)
end

observed_f(col,sol,obstimes,samples,subject) = samples

mstatic = PumasModel(p,rfx_f,col_f,init_f,prob,derived_f,observed_f)

param = init_param(mdsl)
randeffs = init_randeffs(mdsl, param)

subject = data[1]

@test conditional_nll(mdsl,subject,param,randeffs,abstol=1e-12,reltol=1e-12) ≈ conditional_nll(mstatic,subject,param,randeffs,abstol=1e-12,reltol=1e-12)

@test (Random.seed!(1); simobs(mdsl,subject,param,randeffs,abstol=1e-12,reltol=1e-12)[:dv]) ≈
      (Random.seed!(1); simobs(mstatic,subject,param,randeffs,abstol=1e-12,reltol=1e-12)[:dv])


p = ParamSet((θ = VectorDomain(3, lower=zeros(3), init=ones(3)),
              Ω = PSDDomain(2))),

function col_f2(param,randeffs,subject)
  function pre(t)
    (Ka = param.θ[1],
     CL = param.θ[2] * exp(randeffs.η[1]),
     V  = param.θ[3] * exp(randeffs.η[2]))
  end
end

function post_f(col,u,t)
   col_t = col(0.0) # no t needed as covariates are constant
    (conc = u[2] / col_t.V,)
end
function derived_f(col, sol, obstimes, subject, param, randeffs)
  col_t = col(0.0)
  V = col_t.V
  central = sol(obstimes;idxs=2)
  conc = @. central / V
  (conc = conc,)
end

mstatic2 = PumasModel(p,rfx_f,col_f2,init_f,prob,derived_f,observed_f)

subject = Subject(evs = DosageRegimen([10, 20], ii = 24, addl = 2, ss = 1:2, time = [0, 12], cmt = 2))

param = (θ = [
              1.5,  #Ka
              1.0,  #CL
              30.0 #V
              ],
         Ω = Matrix{Float64}(I, 2, 2))
randeffs = (η = zeros(2),)

sol = solve(mstatic2,subject,param,randeffs;obstimes=[i*12+1e-12 for i in 0:1],abstol=1e-12,reltol=1e-12)

p = simobs(mstatic2,subject,param,randeffs;obstimes=[i*12+1e-12 for i in 0:1],abstol=1e-12,reltol=1e-12)
@test 1000p[:conc] ≈ [605.3220736386598;1616.4036675452326]
