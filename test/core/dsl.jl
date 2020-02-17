using Pumas, Test, Random, LabelledArrays


# Read the data# Read the data
data = read_pumas(example_data("data1"), cvs = [:sex,:wt,:etn])
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
        a ∈ ConstDomain(0.2)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @covariates sex wt etn

    @pre begin
        θ1 := θ[1]
        Ka = θ1
        CL = θ[2] * ((wt/70)^0.75) * (θ[4]^sex) * exp(η[1])
        V  = θ[3] * exp(η[2])
    end

    @vars begin
      conc = Central / V
      conc2 = Central^2
    end

    @dynamics begin
        Depot'   := -Ka*Depot # test for `:=` handling
        Central' =  Ka*Depot - CL*conc
    end

    @derived begin
      dv ~ @. Normal(conc, conc*Σ)
      T_max = maximum(t)
    end

    @observed begin
      obs_cmax = maximum(dv)
    end
end

### Function-Based Interface

p = ParamSet((θ = VectorDomain(4, lower=zeros(4), init=ones(4)), # parameters
              Ω = PSDDomain(2),
              Σ = RealDomain(lower=0.0, init=1.0),
              a = ConstDomain(0.2)))

function rfx_f(p)
    ParamSet((η=MvNormal(p.Ω),))
end

function col_f(param,randeffs,subject)
    function f(t=nothing)
        (Ka = param.θ[1],  # pre
        CL = param.θ[2] * ((subject.covariates.wt/70)^0.75) *
             (param.θ[4]^subject.covariates.sex) * exp(randeffs.η[1]),
        V  = param.θ[3] * exp(randeffs.η[2]))
    end
end

OneCompartmentVector = @SLVector (:Depot,:Central)

function init_f(col,t0)
    c = col(t0)
    T = typeof(c.CL/c.V)
    OneCompartmentVector(0.0,0.0)
end

function onecompartment_f(u,p,t)
    OneCompartmentVector(-p.Ka*u[1],
                          p.Ka*u[1] - (p.CL/p.V)*u[2])
end
prob = ODEProblem(onecompartment_f,nothing,nothing,nothing)

# In the function interface, the first return value is a named tuple of sampled
# values, the second is a named tuple of distributions
function derived_f(col,sol,obstimes,subject,  param, randeffs)
    Σ = param.Σ
    V = col().V
    central = sol(obstimes;idxs=2)
    conc = @. central / V
    dv = @. Normal(conc, conc*Σ) # we should move params to a separate output
    (dv=dv,)
end

function observed_f(col,sol,obstimes,samples,subject)
    (obs_cmax = maximum(samples.dv),
     T_max = maximum(obstimes),
     dv = samples.dv)
end

mobj = PumasModel(p,rfx_f,col_f,init_f,prob,derived_f,observed_f)

param = init_param(mdsl)
randeffs = init_randeffs(mdsl, param)

subject = data[1]

sol1 = solve(mdsl,subject,param,randeffs)
sol2 = solve(mobj,subject,param,randeffs)


@test sol1[10].Central ≈ sol2[10].Central
@test sol1[2,:] ≈ sol2[2,:]

sol1 = solve(mdsl,subject,param,randeffs,alg=Rosenbrock23())
sol2 = solve(mobj,subject,param,randeffs,alg=Rosenbrock23())

sol1 = solve(mdsl,subject,param,randeffs,alg=TRBDF2())
sol2 = solve(mobj,subject,param,randeffs,alg=TRBDF2())

sol1 = solve(mdsl,subject,param,randeffs,alg=ROCK2())
sol2 = solve(mobj,subject,param,randeffs,alg=ROCK2())

conditional_nll(mdsl,subject,param,randeffs)
conditional_nll(mobj,subject,param,randeffs)
@test conditional_nll(mdsl,subject,param,randeffs) ≈ conditional_nll(mobj,subject,param,randeffs) rtol=5e-3

Random.seed!(1); obs_dsl = simobs(mdsl,subject,param,randeffs)
Random.seed!(1); obs_obj = simobs(mobj,subject,param,randeffs)

@test obs_dsl.observed.obs_cmax == obs_obj.observed.obs_cmax > 0
@test obs_dsl.observed.T_max == obs_obj.observed.T_max

@test obs_dsl[:dv] ≈ obs_obj[:dv]

# Now test an array-based version

function init_f_iip(col,t0)
    [0.0,0.0]
end

function onecompartment_f_iip(du,u,p,t)
    du[1] = -p.Ka*u[1]
    du[2] =  p.Ka*u[1] - (p.CL/p.V)*u[2]
end
prob = ODEProblem(onecompartment_f_iip,nothing,nothing,nothing)

mobj_iip = PumasModel(p,rfx_f,col_f,init_f_iip,prob,derived_f,observed_f)
sol2 = solve(mobj_iip,subject,param,randeffs)

@test conditional_nll(mobj_iip,subject,param,randeffs) ≈ conditional_nll(mobj,subject,param,randeffs) rtol=5e-3

@test (Random.seed!(1); simobs(mobj_iip,subject,param,randeffs)[:dv]) ≈
      (Random.seed!(1); simobs(mobj,subject,param,randeffs)[:dv]) rtol=1e-4

mdsl = @model begin
    @param begin
        θ ∈ VectorDomain(4, lower=zeros(4), init=ones(4))
        Ω ∈ PSDDomain(2)
        Σ ∈ RealDomain(lower=0.0, init=1.0)
        a ∈ ConstDomain(0.2)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @covariates begin
        sex
        wt
        etn
    end

    @pre begin
        θ1 := θ[1]
        Ka = θ1
        CL = θ[2] * ((wt/70)^0.75) * (θ[4]^sex) * exp(η[1])
        V  = θ[3] * exp(η[2])
    end

    @derived begin
      dv ~ @. Binomial(30, Ka*CL)
    end

    @observed begin
        obs_cmax = maximum(dv)
        T_max = maximum(t)
    end
end
param = init_param(mdsl)
randeffs = init_randeffs(mdsl, param)

@test solve(mdsl,subject,param,randeffs) isa Pumas.NullDESolution
@test simobs(mdsl,subject,param,randeffs) != nothing
@test conditional_nll(mdsl,subject,param,randeffs) == Inf # since real-valued observations
