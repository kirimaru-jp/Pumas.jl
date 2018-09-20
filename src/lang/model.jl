export PKPDModel, init_param, init_random, rand_random, simobs, likelihood, collate

"""
    PKPDModel

A model takes the following arguments
- `param`: a `ParamSet` detailing the parameters and their domain
- `data`: a `Population` object
- `random`: a mapping from a named tuple of parameters -> `RandomEffectSet`
- `collate`: a mapping from the (params, rfx, covars) -> ODE params
- `ode`: an ODE system (either exact or analytical)
- `error`: the error model mapping (param, rfx, data, ode vals) -> sampling dist

The idea is that a user can then do
    fit(model, FOCE)
etc. which would return a FittedModel object

Note:
- we include the data in the model, since they are pretty tightly coupled

Todo:
- auxiliary mappings which don't affect the fitting (e.g. concentrations)
"""
mutable struct PKPDModel{P,Q,R,S,T,V}
    param::P
    random::Q
    collate::R
    init::S
    prob::T
    error::V
    function PKPDModel(param, random, collate, init, ode, error)
        prob = ODEProblem(ODEFunction(ode), nothing, nothing, nothing)
        new{typeof(param), typeof(random),
            typeof(collate), typeof(init),
            DiffEqBase.DEProblem,
            typeof(error)}(param, random, collate, init, prob, error)
    end
end

init_param(m::PKPDModel) = init(m.param)
init_random(m::PKPDModel, param) = init(m.random(param))

"""
    rand_random(m::PKPDModel, param)

Generate a random set of random effects for model `m`, using parameters `param`.
"""
rand_random(m::PKPDModel, param) = rand(m.random(param))


"""
    sol = pkpd_solve(m::PKPDModel, subject::Subject, param,
                     rfx=rand_random(m, param),
                     args...; kwargs...)

Compute the ODE for model `m`, with parameters `param` and random effects
`rfx`. `alg` and `kwargs` are passed to the ODE solver. If no `rfx` are
given, then they are generated according to the distribution determined
in the model.

Returns a tuple containing the ODE solution `sol` and collation `col`.
"""
function DiffEqBase.solve(m::PKPDModel, subject::Subject,
                          param, rfx=rand_random(m, param),
                          args...; kwargs...)
    col = m.collate(param, rfx, subject.covariates)
    _solve(m,subject,col,args...;kwargs...)
end

"""
This internal function is just so that the collation doesn't need to
be repeated in the other API functions
"""
function _solve(m::PKPDModel, subject, col, args...;
                tspan::Tuple{Float64,Float64}=timespan(subject), kwargs...)
  u0  = m.init(col, tspan[1])
  m.prob = remake(m.prob; p=col, u0=u0, tspan=tspan)
  if m.prob.f.f isa ExplicitModel
      return _solve_analytical(m, subject, args...;kwargs...)
  else
      return _solve_diffeq(m, subject, args...;kwargs...)
  end
end

"""
sample(d)

Samples a random value from a distribution or if it's a number assumes it's the
constant distribution and passes it through.
"""
sample(d::Distributions.Sampleable) = rand(d)
sample(d) = d

zval(d) = 0.0
zval(d::Distributions.Normal{T}) where {T} = zero(T)

"""
_lpdf(d,x)

The logpdf. Of a non-distribution it assumes the Dirac distribution.
"""
_lpdf(d,x) = d == x ? 0.0 : -Inf
_lpdf(d::Distributions.Sampleable,x) = logpdf(d,x)

"""
    simobs(m::PKPDModel, subject::Subject, param[, rfx, [args...]];
                  obstimes=observationtimes(subject),kwargs...)

Simulate random observations from model `m` for `subject` with parameters `param` at
`obstimes` (by default, use the times of the existing observations for the subject). If no
`rfx` is provided, then random ones are generated according to the distribution
in the model.
"""
function simobs(m::PKPDModel, subject::Subject, param,
                       rfx=rand_random(m, param),
                       args...; obstimes=observationtimes(subject),continuity=:left,kwargs...)
    col = m.collate(param, rfx, subject.covariates)
    sol = _solve(m, subject, col, args...; kwargs...)
    map(obstimes) do t
        # TODO: figure out a way to iterate directly over sol(t)
        if sol isa PKPDAnalyticalSolution
            errdist = m.error(col,sol(t),t)
        else
            errdist = m.error(col,sol(t,continuity=continuity),t)
        end
        map(sample, errdist)
    end
end

"""
_likelihood(err, obs)

Computes the log-likelihood between the err and obs, only using err terms that
also have observations, and assuming the Dirac distribution for any err terms
that are numbers.
"""
function _likelihood(err::T, obs) where {T}
  syms =  fieldnames(T) ∩ fieldnames(typeof(obs.val))
  sum(map((d,x) -> isnan(x) ? zval(d) : _lpdf(d,x), (getproperty(err,x) for x in syms), (getproperty(obs.val,x) for x in syms)))
end

"""
    likelihood(m::PKPDModel, subject::Subject, param, rfx, args...; kwargs...)

Compute the full log-likelihood of model `m` for `subject` with parameters `param` and
random effects `rfx`. `args` and `kwargs` are passed to ODE solver. Requires that
the post produces distributions.
"""
function likelihood(m::PKPDModel, subject::Subject, param, rfx, args...; kwargs...)
   obstimes = observationtimes(subject)
   col = m.collate(param, rfx, subject.covariates)
   sol = _solve(m, subject, col, args...; kwargs...)
   sum(subject.observations) do obs
       t = obs.time
       err = m.error(col,sol(t),t)
       _likelihood(err, obs)
   end
end

"""
    collate(m::PKPDModel, subject::Subject, param, rfx)

Returns the parameters of the differential equation for a specific subject
subject to parameter and random effects choices. Intended for internal use
and debugging.
"""
function collate(m::PKPDModel, subject::Subject, param, rfx)
   m.collate(param, rfx, subject.covariates)
end
