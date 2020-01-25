using Pumas, Random

# Make sure that PUMASMODELS dict is loaded
if !isdefined(Main, :PUMASMODELS)
  Base.include(Main, "testmodels/testmodels.jl")
end

import Main: PUMASMODELS

@testset "Median size ODE problem (HCV model)" begin
  t = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.99, 10.0, 13.99, 20.99, 28.0]

  peg_inf_model = PUMASMODELS["misc"]["HCV"]["solver"]

  peg_inf_dr = DosageRegimen(180.0, ii=7.0, addl=3, duration=1.0)

  param_PKPD = PUMASMODELS["misc"]["HCV"]["param"]

  _pop = map(i -> Subject(id=i, obs=(yPK=[], yPD=[]), evs=peg_inf_dr, time=t), 1:3)

  # Simulate data for estimation (fix seed for reproducibility)
  Random.seed!(123)
  simdata = simobs(peg_inf_model, _pop, param_PKPD, ensemblealg = EnsembleSerial())

  pd = Subject.(simdata)

  ft = fit(
    peg_inf_model,
    pd,
    param_PKPD,
    Pumas.FOCE(),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=true, x_reltol=1e-3))

  @test deviance(ft) ≈ -100.78347 rtol=1e-5

end
