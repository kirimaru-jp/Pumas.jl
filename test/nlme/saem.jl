using Pumas, Test, CSV, Random, Distributions, TransformVariables, Optim, ForwardDiff

include("Pumas.jl/src/estimation/saem.jl")

theopp = read_pumas(example_data("event_data/THEOPP"),cvs = [:WT,:SEX])

theopmodel_bayes = @model begin
    @param begin
      θ ~ Constrained(MvNormal([1.9,0.0781,0.0463,1.5,0.4],
                               Diagonal([9025, 15.25, 5.36, 5625, 400])),
                      lower=[0.1,0.008,0.0004,0.1,0.0001],
                      upper=[5,0.5,0.09,5,1.5],
                      init=[1.9,0.0781,0.0463,1.5,0.4]
                      )
      Ω ~ InverseWishart(2, fill(0.9,1,1) .* (2 + 1 + 1)) # NONMEM specifies the inverse Wishart in terms of its mode
      σ ∈ RealDomain(lower=0.0, init=0.388)
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      Ka = (SEX == 1 ? θ[1] : θ[4]) + η[1]
      K = θ[2]
      CL  = θ[3]*(WT/70)^θ[5]
      V = CL/K
      SC = V/(WT/70)
    end

    @covariates SEX WT

    @vars begin
        conc = Central / SC
    end

    @dynamics Depots1Central1

    @derived begin
        dv ~ @. Normal(conc,sqrt(σ)+eps())
    end
end

param = Pumas.init_param(theopmodel_bayes)

fit(theopmodel_bayes, theopp, param, SAEM(10), zeros(12))