using Pumas, Test, Random, StaticArrays

# Load data
cvs = [:ka, :cl, :v]
dvs = [:dv]
data = read_pumas(example_data("oral1_1cpt_KAVCL_MD_data"),
                      cvs = cvs, dvs = dvs)

m_diffeq = @model begin

    @covariates ka cl v

    @pre begin
        Ka = ka
        CL = cl
        V = v
    end

    @vars begin
        cp = CL/V
    end

    @dynamics begin
        Depot'   = -Ka*Depot
        Central' =  Ka*Depot - cp*Central
    end

    # we approximate the error by computing the conditional_nll
    @derived begin
        conc = @. Central / V
        dv ~ @. Normal(conc, 1e-100)
    end
end

subject1 = data[1]
param = NamedTuple()
randeffs = NamedTuple()

condition = function (u,t,integrator)
    u.Central - 5e4
end

called = false
tsave = 0.0
affect! = function (integrator)
    global called
    if !called
        called = true
        global tsave = integrator.t
    end
    integrator.u += SA[2e4,5e4]
end

cb = ContinuousCallback(condition,nothing,affect!,save_positions=(false,true))
sol_diffeq   = solve(m_diffeq,subject1,param,randeffs;callback=cb,saveat=72:0.1:300)
@test called
@test all(sol_diffeq[2,sol_diffeq.t .> tsave] .>= 5e4)

condition = function (u,t,integrator)
    u.Central < 5e5
end

called = false
affect! = function (integrator)
    global called = true
    integrator.u += SA[5e5,5e5]
end

cb = DiscreteCallback(condition,affect!,save_positions=(false,true))
sol_diffeq   = solve(m_diffeq,subject1,param,randeffs;callback=cb,saveat=72:0.1:300)
@test called
@test all(sol_diffeq[2,3:end] .>= 2e5)
