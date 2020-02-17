using Pumas, Test, CSV

@testset "nmtran" begin
  data = read_pumas(example_data("event_data/data1"))
  @test_nowarn show(data)

  @test getproperty.(data[1].events, :time) == 0:12:36

  @testset "Subject comparison" begin
    datacopy = deepcopy(data)
    @test data == datacopy
    @test hash(data) == hash(datacopy)
  end

  for ev in data[1].events
    @test ev.amt == data[1].events[1].amt
    @test ev.evid == data[1].events[1].evid
    @test ev.cmt == data[1].events[1].cmt
    @test ev.rate == data[1].events[1].rate
    @test ev.ss == data[1].events[1].ss
    @test ev.ii == data[1].events[1].ii
    @test ev.rate_dir == data[1].events[1].rate_dir
    @test ev.cmt == 1
  end
  @testset "Dosage Regimen" begin
    gen_subject = Subject(evs = DosageRegimen(100, ii = 12, addl = 3))
    @test data[1].id == gen_subject.id
    @test data[1].covariates == gen_subject.covariates
    @test data[1].events == gen_subject.events
    e1 = DosageRegimen(100, ii = 24, addl = 6)
    e2 = DosageRegimen(50,  ii = 12, addl = 13)
    e3 = DosageRegimen(200, ii = 24, addl = 2)
    @test isa(DosageRegimen(e1, e2, e3), DosageRegimen)
  end
end
@testset "Time Variant Covariates" begin
  data = read_pumas(example_data("time_varying_covariates"), cvs = [:weight, :dih])
  @test data[1].covariates.weight |> (x -> isa(x, AbstractVector{Int}) && length(x) == 7)
  @test data[1].covariates.dih == 2
end
@testset "Chronological Observations" begin
  data = DataFrame(time = [0, 1, 2, 2], dv = rand(4), evid = 0)
  @test isa(read_pumas(data), Population)
  append!(data, DataFrame(time = 1, dv = rand(), evid = 0))
  @test_throws AssertionError read_pumas(data)
  @test_throws AssertionError Subject(obs=DataFrame(x=[2:3;], time=1:-1:0))
end
@testset "event_data" begin
  data = DataFrame(time = [0, 1, 2, 2], amt = zeros(4), dv = rand(4), evid = 1)
  @test_throws AssertionError read_pumas(data)
  @test isa(read_pumas(data, event_data = false), Population)
  @test isa(DosageRegimen(100, rate = -2, cmt = 2, ii = 24, addl = 3), DosageRegimen)
  @test isa(DosageRegimen(0, time = 50, evid = 3, cmt = 2), DosageRegimen)
  @test_throws ArgumentError("amt must be 0 for evid = 3") DosageRegimen(1, time = 50, evid = 3, cmt = 2)
  choose_covariates() = (isPM = rand(["yes", "no"]), Wt = rand(55:80))
  generate_population(events, nsubs = 24) =
    Population([ Subject(id = i, evs = events, cvs = choose_covariates()) for i ∈ 1:nsubs ])
  firstdose = DosageRegimen(100, ii = 12, addl = 3, rate = 50)
  seconddose = DosageRegimen(0, time = 50, evid = 3, cmt = 2)
  thirddose = DosageRegimen(120, time = 54, ii = 16, addl = 2)
  ev = reduce(DosageRegimen, [firstdose, seconddose, thirddose])
  @test isa(generate_population(ev), Population)
end
@testset "Population Constructors" begin
  e1 = DosageRegimen(100, ii = 24, addl = 6)
  e2 = DosageRegimen(50,  ii = 12, addl = 13)
  e3 = DosageRegimen(200, ii = 24, addl = 2)
  s1 = map(i -> Subject(id = i, evs = e1), 1:5)
  s2 = map(i -> Subject(id = i, evs = e2), 6:8)
  s3 = map(i -> Subject(id = i, evs = e3), 9:10)
  pop1 = Population(s1)
  pop2 = Population(s2)
  pop3 = Population(s3)
  @test Population(s1, s2, s3) == Population(pop1, pop2, pop3)
end
@testset "Missing Observables" begin
  data = read_pumas(DataFrame(time = [0, 1, 2, 4],
                                  evid = [1, 1, 0, 0],
                                  dv1 = [0, 0, 1, missing],
                                  dv2 = [0, 0, missing, 2],
                                  x = [missing, missing, 0.5, 0.75],
                                  amt = [0.25, 0.25, 0, 0]),
                         cvs = [:x],
                         dvs = [:dv1, :dv2])
  @test isa(data, Population)
  isa(data[1].observations[1], NamedTuple{(:dv1,:dv2),NTuple{2, Union{Missing,Float64}}})
end
@testset "DataFrames Constructors" begin
  e1 = DosageRegimen(100, ii = 24, addl = 6)
  e2 = DosageRegimen(50, ii = 12, addl = 13)
  e3 = DosageRegimen(200, ii = 24, addl = 2)
  evs = reduce(DosageRegimen, [e1, e2, e3])
  data = DataFrame(evs, true)
  @test size(data, 1) == 24
end

@testset "MDV" begin
  data = DataFrame(amt = 10, dv = 0, evid = 0, mdv = 1)
  output = read_pumas(data)
  @test ismissing(output[1].observations.dv[1])
end

@testset "amt = rate * duration" begin
  e1 = DosageRegimen(100, rate = 25)
  e2 = DosageRegimen(100, duration = 4)
  @test e1.data == e2.data
  @test_throws AssertionError DosageRegimen(100, duration = 4, rate = 20)
end
