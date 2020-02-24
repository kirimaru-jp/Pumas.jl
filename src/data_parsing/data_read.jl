"""
  to_nt(obj)::NamedTuple{PN,VT}

It returns a NamedTuple based on the propertynames of the object.
If a value is a vector with a single value, it returns the value.
If the vector has no missing values, it is promoted through disallowmissing.
"""
to_nt(obj::Any) = propertynames(obj) |>
  (x -> NamedTuple{Tuple(x)}(
    getproperty(obj, x) |>
    (x -> isone(length(unique(x))) ?
          first(x) :
          x)
    for x ∈ x))

"""
    read_pumas(filepath::String, args...; kwargs...)
    read_pumas(data; dvs=Symbol[:dv], cvs=Symbol[],
                   id=:id, time=:time, evid=:evid,
                   amt=:amt, addl=:addl, ii=:ii, cmt=:cmt,
                   rate=:rate, ss=:ss,
                   event_data = true)

Import PREDPP-formatted data.

- `dvs` dependent variables specified by column names
- `cvs` covariates specified by column names
- `event_data` toggles assertions applicable to event data
"""
function read_pumas(filepath::AbstractString; kwargs...)
  read_pumas(CSV.read(filepath, missingstrings=["."]) ; kwargs...)
end
function read_pumas(df::DataFrame;
  dvs::Vector{Symbol}         = Symbol[:dv],
  cvs::Vector{Symbol}         = Symbol[],
  id::Symbol                  = :id,
  time::Symbol                = :time,
  evid::Symbol                = :evid,
  mdv::Union{Symbol,Nothing}  = nothing,
  amt::Union{Symbol,Nothing}  = nothing,
  addl::Union{Symbol,Nothing} = nothing,
  ii::Union{Symbol,Nothing}   = nothing,
  cmt::Union{Symbol,Nothing}  = nothing,
  rate::Union{Symbol,Nothing} = nothing,
  ss::Union{Symbol,Nothing}   = nothing,
  event_data::Bool            = true)

  _df = copy(df)
  colnames = names(_df)

  # Ensure that dv columns allow for missing values
  allowmissing!(_df, dvs)

  # We'll require that id, time, and evid available in the dataset
  for invar in (id, time, evid)
    DataFrames.lookupname(DataFrames.index(df).lookup, invar)
  end

  # For mdv, amt, addl, ii, cmt, rate, and ss we'll use default values if
  # the user hasn't specified names. If the user has specified names,
  # we check that the name exists in the input DataFrame
  if amt === nothing
    _amt = :amt
  else
    DataFrames.lookupname(DataFrames.index(df).lookup, amt)
    _amt = amt
  end
  if addl === nothing
    _addl = :addl
  else
    DataFrames.lookupname(DataFrames.index(df).lookup, addl)
    _addl = addl
  end
  if ii === nothing
    _ii = :ii
  else
    DataFrames.lookupname(DataFrames.index(df).lookup, ii)
    _ii = ii
  end
  if cmt === nothing
    _cmt = :cmt
  else
    DataFrames.lookupname(DataFrames.index(df).lookup, cmt)
    _cmt = cmt
  end
  if rate === nothing
    _rate = :rate
  else
    DataFrames.lookupname(DataFrames.index(df).lookup, rate)
    _rate = rate
  end
  if ss === nothing
    _ss = :ss
  else
    DataFrames.lookupname(DataFrames.index(df).lookup, ss)
    _ss = ss
  end

  # Missing variables require special care
  # Internally in Pumas, we use missing values in the dv column to encode missings
  if mdv !== nothing
    # If mdv has been set then we check that the column exists
    DataFrames.lookupname(DataFrames.index(df).lookup, mdv)
    for dv in dvs
      _df[!, dv] .= ifelse.(_df[!, mdv] .== 1, missing, _df[!, dv])
    end
  else
    # Default name for missing values column is mdv
    if :mdv ∈ colnames
      for dv in dvs
        _df[!, dv] .= ifelse.(_df.mdv .== 1, missing, _df[!, dv])
      end
    end
  end

  return [Subject(_databyid, colnames, id, time, evid, _amt, _addl, _ii, _cmt,
                  _rate, _ss, cvs, dvs, event_data) for _databyid in groupby(_df, id)]
end

function build_observation_list(obs::AbstractDataFrame)
  #cmt = :cmt ∈ names(obs) ? obs[:cmt] : 1
  vars = setdiff(names(obs), (:time, :cmt))
  return NamedTuple{ntuple(i->vars[i],length(vars))}(ntuple(i -> convert(AbstractVector{Union{Missing,Float64}}, obs[!,vars[i]]), length(vars)))
end
build_observation_list(obs::NamedTuple) = obs
build_observation_list(obs::Nothing) = obs

build_event_list(evs::AbstractVector{<:Event}, event_data::Bool) = evs
function build_event_list!(events, event_data, t, evid, amt, addl, ii, cmt, rate, ss)
  @assert evid ∈ 0:4 "evid must be in 0:4"
  # Dose-related data items
  drdi = iszero(amt) && (rate == 0) && iszero(ii) && iszero(addl) && iszero(ss)
  if event_data
    if evid ∈ [0, 2, 3]
      @assert drdi "Dose-related data items must be zero when evid = $evid"
    else
      @assert !drdi "Some dose-related data items must be non-zero when evid = $evid"
    end
  end
  duration = amt / rate
  for j = 0:addl  # addl==0 means just once
    _ss = iszero(j) ? ss : zero(Int8)
    if iszero(amt) && evid ≠ 2
      # These are dose events having AMT=0, RATE>0, SS=1, and II=0.
      # Such an event consists of infusion with the stated rate,
      # starting at time −∞, and ending at the time on the dose
      # ev event record. Bioavailability fractions do not apply
      # to these doses.
      push!(events, Event(amt, t, evid, cmt, rate, ii, _ss, ii, t, Int8(1)))
    else
      push!(events, Event(amt, t, evid, cmt, rate, duration, _ss, ii, t, Int8(1)))
      if !iszero(rate) && iszero(_ss)
        push!(events, Event(amt, t + duration, Int8(-1), cmt, rate, duration, _ss, ii, t, Int8(-1)))
      end
    end
    t += ii
  end
end
function build_event_list(regimen::DosageRegimen, event_data::Bool)
  data = regimen.data
  events = Event[]
  for i in 1:size(data, 1)
    t    = data[!,:time][i]
    evid = data[!,:evid][i]
    amt  = data[!,:amt][i]
    addl = data[!,:addl][i]
    ii   = data[!,:ii][i]
    cmt  = data[!,:cmt][i]
    rate = data[!,:rate][i]
    ss   = data[!,:ss][i]
    build_event_list!(events, event_data, t, evid, amt, addl, ii, cmt, rate, ss)
  end
  sort!(events)
end
