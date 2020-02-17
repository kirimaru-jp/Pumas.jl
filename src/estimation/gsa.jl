function DiffEqSensitivity.gsa(m::PumasModel, subject::Subject, params::NamedTuple, method::DiffEqSensitivity.GSAMethod, vars = [:dv], p_range_low=NamedTuple{keys(params)}([par.*0.05 for par in values(params)]), p_range_high=NamedTuple{keys(params)}([par.*1.95 for par in values(params)]), args...; kwargs...)
    vlowparam = TransformVariables.inverse(toidentitytransform(m.param), p_range_low)
    vhighparam = TransformVariables.inverse(toidentitytransform(m.param), p_range_high)
    p_range = [[vlowparam[i], vhighparam[i]] for i in 1:length(vlowparam)]

    trf_ident = toidentitytransform(m.param)

    sim_ = simobs(m, subject, params, args...; kwargs...)
    length_vars = [length(sim_.observed[key]) for key in vars]

    function f(p)
        param = TransformVariables.transform(trf_ident, p)
        sim = simobs(m, subject, param, args...; kwargs...)
        collect(Iterators.flatten([sim.observed[key] for key in vars])) 
    end

    sensitivity = DiffEqSensitivity.gsa(f, method, p_range; kwargs...)

    return sens_result(sensitivity, p_range_low, vars, length_vars)
end

function DiffEqSensitivity.gsa(m::PumasModel, population::Population, params::NamedTuple, method::DiffEqSensitivity.GSAMethod, vars = [:dv], p_range_low=NamedTuple{keys(params)}([par.*0.05 for par in values(params)]), p_range_high=NamedTuple{keys(params)}([par.*1.95 for par in values(params)]), args...; kwargs...)
    vlowparam = TransformVariables.inverse(toidentitytransform(m.param), p_range_low)
    vhighparam = TransformVariables.inverse(toidentitytransform(m.param), p_range_high)
    p_range = [[vlowparam[i], vhighparam[i]] for i in 1:length(vlowparam)]

    trf_ident = toidentitytransform(m.param)

    sim_ = simobs(m, population, params, args...; kwargs...)
    length_vars = [length(sim_[1].observed[key]) for key in vars]

    function f(p)
        param = TransformVariables.transform(trf_ident, p)
        sim = simobs(m, population, param, args...; kwargs...)
        mean([collect(Iterators.flatten([sim[i].observed[key] for key in vars])) for i in 1:length(sim)])
    end

    sensitivity = DiffEqSensitivity.gsa(f, method, p_range; kwargs...)

    return sens_result(sensitivity, p_range_low, vars, length_vars)
end

struct SobolOutput{T1, T2, T3, T4, T5, T6}
    first_order::T1
    total_order::T2
    second_order::T3
    first_order_conf_int::T4
    total_order_conf_int::T5
    second_order_conf_int::T6
end

function sens_result(sens::DiffEqSensitivity.SobolResult, p_range_low::NamedTuple, vars::AbstractVector, length_vars::AbstractVector)
    s1 = sens.S1
    st = sens.ST
    s2 = sens.S2 === nothing ? nothing : sens.S2

    var_name = []
    for i in 1:length(vars)
        len_var = length_vars[i]
        if len_var > 1
            append!(var_name,[string(vars[i], "[$j]") for j in 1:len_var])
        else
            push!(var_name,string(vars[i]))
        end
    end

    par_name = []
    for i in 1:length(p_range_low)
        len_par = length(p_range_low[i])
        if len_par > 1
            append!(par_name, [string(keys(p_range_low)[i],"$j") for j in 1:len_par])
        else
            push!(par_name, string(keys(p_range_low)[i]))
        end
    end

    S1 = DataFrame(dv_name = var_name)
    ST = DataFrame(dv_name = var_name)

    for i in 1:length(par_name)
        insertcols!(S1, i+1, Symbol(par_name[i]) => s1[:,i])
        insertcols!(ST, i+1, Symbol(par_name[i]) => st[:,i])
    end
    if s2 != nothing
        par_name_sec = [string(par_name[i],"*" ,par_name[j]) for i in 1:length(par_name) for j in i+1:length(par_name)]
        S2 = DataFrame(dv_name = var_name)
        for i in 1:length(par_name_sec)
            insertcols!(S2, i+1, Symbol(par_name_sec[i]) => s2[:,i])
        end
    end

    return SobolOutput(S1, ST, s2 === nothing ? nothing : S2 , nothing, nothing, nothing)
end


struct MorrisOutput{T1, T2, T3}
    means::T1
    means_star::T2
    variances::T3
end

function sens_result(sens::DiffEqSensitivity.MorrisResult, p_range_low::NamedTuple, vars::AbstractVector, length_vars::AbstractVector)
    means = sens.means
    means_star = sens.means_star
    variances = sens.variances

    var_name = []
    for i in 1:length(vars)
        len_var = length_vars[i]
        if len_var > 1
            append!(var_name,[string(vars[i], "[$j]") for j in 1:len_var])
        else
            push!(var_name,string(vars[i]))
        end
    end

    par_name = []
    for i in 1:length(p_range_low)
        len_par = length(p_range_low[i])
        if len_par > 1
            append!(par_name, [string(keys(p_range_low)[i],"$j") for j in 1:len_par])
        else
            push!(par_name, string(keys(p_range_low)[i]))
        end
    end

    μ = DataFrame(dv_name = var_name)
    μ_star = DataFrame(dv_name = var_name)
    variance = DataFrame(dv_name = var_name)

    for i in 1:length(par_name)
        insertcols!(μ, i+1, Symbol(par_name[i]) => means[:,i])
        insertcols!(μ_star, i+1, Symbol(par_name[i]) => means_star[:,i])
        insertcols!(variance, i+1, Symbol(par_name[i]) => variances[:,i])
    end

    return MorrisOutput(μ, μ_star, variance)
end

struct eFASTOutput{T1, T2}
    first_order::T1
    total_order::T2
end

function sens_result(sens::DiffEqSensitivity.eFASTResult, p_range_low::NamedTuple, vars::AbstractVector, length_vars::AbstractVector)
    s1 = sens.first_order
    st = sens.total_order

    var_name = []
    for i in 1:length(vars)
        len_var = length_vars[i]
        if len_var > 1
            append!(var_name,[string(vars[i], "[$j]") for j in 1:len_var])
        else
            push!(var_name,string(vars[i]))
        end
    end

    par_name = []
    for i in 1:length(p_range_low)
        len_par = length(p_range_low[i])
        if len_par > 1
            append!(par_name, [string(keys(p_range_low)[i],"$j") for j in 1:len_par])
        else
            push!(par_name, string(keys(p_range_low)[i]))
        end
    end

    S1 = DataFrame(dv_name = var_name)
    ST = DataFrame(dv_name = var_name)

    for i in 1:length(par_name)
        insertcols!(S1, i+1, Symbol(par_name[i]) => s1[:,i])
        insertcols!(ST, i+1, Symbol(par_name[i]) => st[:,i])
    end

    return eFASTOutput(S1, ST)
end
