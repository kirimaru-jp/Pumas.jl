using Pumas, LinearAlgebra, Test

# Make sure that PUMASMODELS dict is loaded
if !isdefined(Main, :PUMASMODELS)
  Base.include(Main, "testmodels/testmodels.jl")
end

import Main: PUMASMODELS

@testset "Test informationmatrix with warfarin data" begin

  warfarin = PUMASMODELS["1cpt"]["oral"]["normal_additive"]["warfarin"]
  data, model, param = warfarin["data"], warfarin["analytical"], warfarin["param"]

  @test logdet(
    sum(
      Pumas._expected_information(
        model,
        d,
        param,
        Pumas._orth_empirical_bayes(model, d, param, Pumas.FO()),
        Pumas.FO()
      ) for d in data)) ≈ 53.8955 rtol=1e-6

  ft = fit(model, data, param, Pumas.FO())

  @test logdet(informationmatrix(ft)) isa Number

end

@testset "Multiple dvs. (The HCV model)" begin

  hcv = PUMASMODELS["misc"]["HCV"]
  peg_inf_model, param_PKPD = hcv["solver"], hcv["param"]

  t = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 7.0, 10.0, 14.0, 21.0, 28.0]
  
  peg_inf_dr = DosageRegimen(180.0, ii=7.0, addl=3, duration=1.0)
  _sub = Subject(id=1, evs=peg_inf_dr, time=t, obs=(yPK=zeros(length(t)), yPD=zeros(length(t))))

  @test logdet(Pumas._expected_information_fd(peg_inf_model, _sub, param_PKPD, zeros(7), Pumas.FO())*30) ≈ 92.21128100630904 rtol=1e-6

end
