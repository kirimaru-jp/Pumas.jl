using Pumas, Test, CSV, Random, Distributions, TransformVariables, Optim, ForwardDiff

include("../../src/estimation/saem.jl")

data = read_pumas(example_data("sim_data_model1"))

mdsl1 = @model begin
    @param begin
        θ ∈ VectorDomain(2, init=[0.5,1.0])
        Ω ∈ ConstDomain(Diagonal([0.04]))
        Σ ∈ ConstDomain(0.1)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        CL = θ[1] * exp(η[1])
        V  = θ[2]
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

sim = simobs(mdsl1, data, param)
df = DataFrame(sim)
data = read_pumas(df,time=:time)
res = fit(mdsl1,data,param,Pumas.FOCEI())
baye = fit(mdsl1, data, param, SAEM(10), zeros(12),500)
