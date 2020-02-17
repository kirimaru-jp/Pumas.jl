using Test
using Pumas

@testset "args and kwargs" begin
data = read_pumas(example_data("sim_data_model1"))
#-----------------------------------------------------------------------# Test 1
mdsl1 = @model begin
    @param begin
        θ ∈ VectorDomain(1, init=[0.5])
        Ω ∈ ConstDomain(Diagonal([0.04]))
        Σ ∈ ConstDomain(0.1)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        CL = θ[1] * exp(η[1])
        V  = 1.0
    end

    @vars begin
        conc = Central / V
    end

    @dynamics Central1

    @derived begin
        dv ~ @. Normal(conc,conc*sqrt(Σ)+eps())
    end
end

param = init_param(mdsl1)

ft_no_args = fit(mdsl1, data, param, Pumas.FOCEI())
@test isempty(pairs(ft_no_args.args))
@test isempty(pairs(ft_no_args.kwargs))

ft_alg_kwargs = fit(mdsl1, data, param, Pumas.FOCEI(); alg=Rosenbrock23())
@test isempty(pairs(ft_alg_kwargs.args))
kwarg_pairs = pairs(ft_alg_kwargs.kwargs)
@test keys(kwarg_pairs) == (:alg,)
@test kwarg_pairs[1] == Rosenbrock23()

end

@testset "dv names" begin
     df = DataFrame(read_pumas(example_data("sim_data_model1")))
     df[!, :dv_conc] = df.dv
     data = read_pumas(df; dvs=[:dv_conc])
     df_missing = DataFrame(read_pumas(example_data("sim_data_model1")))
     df_missing[!, :dv_conc] = df_missing.dv
     data_missing = read_pumas(df_missing; dvs=[:dv_conc])
     # Make a missing observation
     push!(data_missing[1].observations.dv_conc, missing)
     push!(data_missing[1].time, 2)

     model = Dict()

     model["additive"] = @model begin
       @param begin
           θ ∈ RealDomain()
           Ω ∈ PDiagDomain(1)
           σ ∈ RealDomain(lower=0.0001)
       end

       @random begin
           η ~ MvNormal(Ω)
       end

       @pre begin
           CL = θ * exp(η[1])
           V  = 1.0
       end

       @vars begin
           conc = Central / V
       end

       @dynamics Central1

       @derived begin
           dv_conc ~ @. Normal(conc, σ)
       end
     end

     model["proportional"] = @model begin
       @param begin
           θ ∈ RealDomain()
           Ω ∈ PDiagDomain(1)
           σ ∈ RealDomain(lower=0.0001)
       end

       @random begin
           η ~ MvNormal(Ω)
       end

       @pre begin
           CL = θ * exp(η[1])
           V  = 1.0
       end

       @vars begin
           conc = Central / V
       end

       @dynamics Central1

       @derived begin
           dv_conc ~ @. Normal(conc, conc*σ)
       end
     end

     model["exponential"] = @model begin
       @param begin
           θ ∈ RealDomain()
           Ω ∈ PDiagDomain(1)
           σ ∈ RealDomain(lower=0.0001)
       end

       @random begin
           η ~ MvNormal(Ω)
       end

       @pre begin
           CL = θ * exp(η[1])
           V  = 1.0
       end

       @vars begin
           conc = Central / V
       end

       @dynamics Central1

       @derived begin
           dv_conc ~ @. LogNormal(log(conc), σ)
       end
     end

     param = (θ=0.5, Ω=Diagonal([0.04]), σ=0.01)

     @testset "testing model: $_model, with $_approx approximation" for
       _model in ("additive", "proportional", "exponential"),
         _approx in (Pumas.FO(), Pumas.FOCE(), Pumas.FOCEI(), Pumas.LaplaceI())

       if _model == "proportional" && _approx == Pumas.FOCE()
        @test_throws ArgumentError deviance(fit(model[_model], data, param, _approx))
        continue
      end
       # LaplaceI and proportional is very unstable and succeeds/fails depending on architecture
       # so we can't mark this as @test_broken
       if _model != "proportional" || _approx != Pumas.LaplaceI()
           @test deviance(fit(model[_model], data, param, _approx)) == deviance(fit(model[_model], data_missing, param, _approx))
           res = fit(model[_model], data, param, _approx)
           @test Pumas.∂²l∂η²(model[_model], first(data), param, first(res.vvrandeffsorth), _approx) isa Tuple
       end
     end
end # begin

@testset "two independent dvs" begin
    theopp = read_pumas(example_data("event_data/THEOPP"),cvs = [:SEX,:WT])
    theopp_df = DataFrame(theopp)
    theopp_df[!,:dv2] = theopp_df[!,:dv]
    theoppnew = read_pumas(theopp_df; dvs=[:dv, :dv2], cvs=[:SEX, :WT])
    theopmodel_solver_fo = @model begin
      @param begin
        θ₁ ∈ RealDomain(lower=0.1,    upper=5.0, init=2.77)
        θ₂ ∈ RealDomain(lower=0.0008, upper=0.5, init=0.0781)
        θ₃ ∈ RealDomain(lower=0.004,  upper=0.9, init=0.0363)
        θ₄ ∈ RealDomain(lower=0.1,    upper=5.0, init=1.5)
        Ω ∈ PSDDomain(3)
        σ_add ∈ RealDomain(lower=0.001, init=0.388)
        #σ_prop ∈ RealDomain(init=0.3)
      end

      @random begin
        η ~ MvNormal(Ω)
      end

      @pre begin
        Ka = SEX == 0 ? θ₁ + η[1] : θ₄ + η[1]
        K  = θ₂+ η[2]
        CL = θ₃*WT + η[3]
        V  = CL/K
        SC = CL/K/WT
      end

      @covariates SEX WT

      @vars begin
        conc = Central / SC
        cp   = Central/V
      end

      @dynamics begin
          Depot'   = -Ka*Depot
          Central' =  Ka*Depot - (CL/V)*Central
      end

      @derived begin
        dv ~ @. Normal(conc,sqrt(σ_add))
        dv2 ~ @. Normal(conc,sqrt(σ_add))
      end
    end

    param = (
      θ₁ = 2.77,   #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
      θ₂ = 0.0781, #K MEAN ELIMINATION RATE CONSTANT (1/HR)
      θ₃ = 0.0363, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
      θ₄ = 1.5,    #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)
      Ω  = diagm(0 => [5.55, 0.0024, 0.515]), # update to block diagonal
      σ_add = 0.388
      # σ_prop = 0.3
         )
     o = fit(theopmodel_solver_fo, theopp, param, Pumas.FO())
     o = fit(theopmodel_solver_fo, theoppnew, param, Pumas.FO())
end # begin
