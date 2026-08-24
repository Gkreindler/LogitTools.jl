#=
Worked example: multinomial logit with an OPTION-LEVEL random effect.

The setup is a ranked / repeated-choice panel. Each person meets a menu of
alternatives several times, and we want two different kinds of unobserved
heterogeneity:

  1. a taste for an ATTRIBUTE that the person carries into every choice
     -> a random coefficient at the person level, which is what logit2_rfx does

  2. a taste for a particular ALTERNATIVE, shared across the choices in which
     that person meets it, and different from their taste for the alternative
     next to it
     -> a random INTERCEPT at the (person, alternative) level, which is the
        thing that needs mlogit_rfx

Run with:
    julia --project=. examples/example_mlogit_rfx.jl
=#

using Random
using Distributed
using DataFrames
using Statistics
using StatsBase
using Distributions
using Optim
using LogitTools

# ---------------------------------------------------------------------------
# 1. Simulate
# ---------------------------------------------------------------------------

const N_PEOPLE = 400      # individuals               (the integration unit)
const N_SETS   = 6        # choice sets per person    (the softmax groups)
const N_OPTS   = 4        # options offered per set
const N_ALTS   = 6        # alternatives in the world (the rfx level)

const BETA    = [0.8, -0.5, 0.3]
const SIGMA_G = 0.7       # true sd of the person-level coefficient on x1
const SIGMA_O = 0.9       # true sd of the (person, alternative) random effect

function simulate(; seed = 2024, sigma_g = SIGMA_G, sigma_o = SIGMA_O)
    rng = MersenneTwister(seed)

    nu = randn(rng, N_PEOPLE)              # person-level taste shifter
    xi = randn(rng, N_PEOPLE, N_ALTS)      # person x alternative taste shifter

    rows = NamedTuple[]
    setid = 0
    for i in 1:N_PEOPLE, s in 1:N_SETS
        setid += 1
        for a in randperm(rng, N_ALTS)[1:N_OPTS]
            rows = push!(rows, (uniqueid = i, setid = setid, alt = a,
                                x1 = randn(rng), x2 = randn(rng), x3 = randn(rng),
                                nu = nu[i], xi = xi[i, a]))
        end
    end
    df = DataFrame(rows)

    df.v = BETA[1] .* df.x1 .+ BETA[2] .* df.x2 .+ BETA[3] .* df.x3 .+
           sigma_g .* df.nu .* df.x1 .+
           sigma_o .* df.xi

    # each person picks the utility-maximising option of each set
    df.selected = zeros(Float64, nrow(df))
    for g in groupby(df, :setid)
        g.selected[argmax(g.v .+ rand(rng, Gumbel(), nrow(g)))] = 1.0
    end

    return select(df, Not([:v, :nu, :xi]))
end

df = simulate()

println("="^76)
println("DATA")
println("="^76)
println("  rows        : ", nrow(df))
println("  individuals : ", length(unique(df.uniqueid)), "   (col_group)")
println("  choice sets : ", length(unique(df.setid)), "   (col_id)")
println("  alternatives: ", length(unique(df.alt)), "   (the rfx level)")
println("  true beta   : ", BETA)
println("  true sigmas : sd(coef on x1) = $SIGMA_G, sd(person x alt effect) = $SIGMA_O")

const MYXS = [:x1, :x2, :x3]
const OPTS = Optim.Options(iterations = 100_000, g_tol = 1e-6)

# ---------------------------------------------------------------------------
# 2. Plain mlogit -- the starting point, and the comparison
# ---------------------------------------------------------------------------

println()
println("="^76)
println("PLAIN mlogit (no heterogeneity: misspecified on purpose)")
println("="^76)

plain = mlogit(df, MYXS, :setid, :selected, zeros(length(MYXS));
               optim_options = OPTS)
plain.converged || error("plain mlogit did not converge")

println("  beta_hat = ", round.(plain.theta_hat, digits = 3))
println("  beta_true= ", BETA)
println()
println("  Note the attenuation towards zero. Ignoring an option-level random")
println("  effect does not just cost you the variance parameter -- it biases the")
println("  slopes, because the omitted effect is correlated with nothing but")
println("  still inflates the residual scale.")

# ---------------------------------------------------------------------------
# 3. mlogit_rfx
# ---------------------------------------------------------------------------

println()
println("="^76)
println("mlogit_rfx: person-level slope on x1 + option-level intercept on :alt")
println("="^76)

myrfx = [:x1,                          # person-level random coefficient
         rfx_term(level = :alt)]       # (person, alternative) random intercept

th0 = theta0_mlogit_rfx(MYXS, myrfx; b0 = plain.theta_hat, col_group = :uniqueid)
println("  theta0 = ", round.(th0, digits = 3))

t = @elapsed fit = mlogit_rfx(df, MYXS, :setid, :selected, th0;
                              col_group = :uniqueid,
                              rfx       = myrfx,
                              ndraws    = 400,
                              optim_options = OPTS)
fit.converged || error("mlogit_rfx did not converge")

println("  fitted in $(round(t, digits = 1))s, $(fit.iterations) iterations")
println()
println("  ", rpad("parameter", 14), rpad("estimate", 12), "truth")
for (j, nm) in enumerate(fit.theta_names)
    truth = j <= 3 ? BETA[j] : (j == 4 ? SIGMA_G : SIGMA_O)
    println("  ", rpad(nm, 14), rpad(round(fit.theta_hat[j], digits = 3), 12), truth)
end

# ---------------------------------------------------------------------------
# 4. Diagnostics -- read these BEFORE the estimates
# ---------------------------------------------------------------------------

println()
println("="^76)
println("DIAGNOSTICS")
println("="^76)

e = fit.extra
println("  ESS (R = $(e.R)): min $(round(e.ess_min, digits = 0)) / " *
        "p10 $(round(e.ess_p10, digits = 0)) / median $(round(e.ess_median, digits = 0))")
println()
println("  cell structure (rfx_cell_report):")
rep = rfx_cell_report(fit)
show(stdout, rep[:, [:term, :level, :n_cells, :cells_per_group_max,
                     :rows_per_cell_median, :sets_per_cell_median,
                     :cancel_share]]; allcols = true)
println()
println()
println("  cells_per_group_max SUMMED OVER THE TERMS is the dimension of the")
println("  simulated integral -- here $(sum(rep.cells_per_group_max)). Read it next to")
println("  ess_p10: a big dimension with a small ESS means the draw set is too")
println("  thin for the model, not that the model is wrong.")

# ---------------------------------------------------------------------------
# 5. Is sigma real, or is it simulation noise?
# ---------------------------------------------------------------------------

println()
println("="^76)
println("SEED STABILITY -- the check the bootstrap cannot do for you")
println("="^76)
println("  Draws are fixed across bootstrap replicates (they must be, or the")
println("  objective is not deterministic in theta), so simulation error never")
println("  enters boot_se. Refitting at other seeds is the only way to see it.")
println()

for R in (200, 400)
    ss = Float64[]
    for sd in (20260808, 111, 222, 333, 444)
        f = mlogit_rfx(df, MYXS, :setid, :selected, th0;
                       col_group = :uniqueid, rfx = myrfx, ndraws = R,
                       seed = sd, optim_options = OPTS)
        push!(ss, f.theta_hat[5])
    end
    println("  R = $(lpad(R, 4)) : sd_1|alt over $(length(ss)) seeds = ",
            round.(ss, digits = 4), "   seed-to-seed sd = ",
            round(std(ss), digits = 4))
end
println()
println("  Monte Carlo error falls like 1/sqrt(R). If the spread does NOT fall,")
println("  the parameter is weakly identified and more draws will not help.")
println()
println("  Caveat: a handful of seeds estimates that spread very imprecisely, so")
println("  read a non-monotone pattern at 5 seeds as inconclusive rather than as")
println("  evidence of weak identification. Use more seeds before concluding.")

# ---------------------------------------------------------------------------
# 6. Bootstrap and report
# ---------------------------------------------------------------------------

println()
println("="^76)
println("BOOTSTRAP (small nboot, for a runnable example)")
println("="^76)
println("  For real work use nboot = 500 and parallel = true:")
println("      using Distributed; addprocs(12); @everywhere using LogitTools")
println()

fit.vcov = boot_mlogit_rfx(df, MYXS, :setid, :selected, th0;
                           col_group   = :uniqueid,
                           rfx         = myrfx,
                           ndraws      = 400,
                           nboot       = 40,
                           cluster_var = :uniqueid,
                           theta_start = fit.theta_hat,   # warm start: much faster
                           parallel    = nprocs() > 1)

nok = count(f -> f.converged && !f.errored, fit.vcov.boot_fits)
println("  $nok / 40 replicates converged")
println()

report = boot_report(fit)
show(stdout, report; allcols = true)
println()

println()
println("="^76)
println("TABLE")
println("="^76)
regtable_rfx(fit;
             labels = Dict("x1" => "X1", "x2" => "X2", "x3" => "X3",
                           "sd_x1"    => "sd: taste for X1 (person)",
                           "sd_1|alt" => "sd: taste for alternative (person x alt)"),
             digits = 3, digits_stats = 3,
             extralines = [["Simulation draws", string(fit.extra.R)],
                           ["Random coefficients", "2"]]) |> display

# ---------------------------------------------------------------------------
# 7. A placebo
# ---------------------------------------------------------------------------

println()
println("="^76)
println("PLACEBO: the same estimator on data with NO option-level effect")
println("="^76)

df0 = simulate(seed = 31, sigma_o = 0.0)
p0  = mlogit(df0, MYXS, :setid, :selected, zeros(length(MYXS)); optim_options = OPTS)
th0b = theta0_mlogit_rfx(MYXS, myrfx; b0 = p0.theta_hat, col_group = :uniqueid)
f0 = mlogit_rfx(df0, MYXS, :setid, :selected, th0b;
                col_group = :uniqueid, rfx = myrfx, ndraws = 400, optim_options = OPTS)

println("  sd_1|alt = ", round(f0.theta_hat[5], digits = 3), "   (truth 0)")
println("  sd_x1    = ", round(f0.theta_hat[4], digits = 3), "   (truth $SIGMA_G)")
println()
println("  A small positive number rather than 0 is expected: sigma is")
println("  canonicalised with abs(), so noise around a true zero always reads")
println("  positive. That is why such a row should be read off its percentile")
println("  interval as \"cannot reject homogeneity\", not as a small effect.")

# ---------------------------------------------------------------------------
# 8. The guard that catches the most common mistake
# ---------------------------------------------------------------------------

println()
println("="^76)
println("GUARD: a random intercept at the PERSON level is not identified here")
println("="^76)
try
    mlogit_rfx(df, MYXS, :setid, :selected, vcat(plain.theta_hat, [0.5]);
               col_group = :uniqueid,
               rfx = [rfx_term(level = :uniqueid)],   # <- shifts every option equally
               ndraws = 64)
catch err
    println("  ", replace(sprint(showerror, err), "\n" => "\n  "))
end
