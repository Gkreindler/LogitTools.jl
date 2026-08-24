using Documenter
using LogitTools

DocMeta.setdocmeta!(LogitTools, :DocTestSetup, :(using LogitTools); recursive = true)

makedocs(
    sitename = "LogitTools.jl",
    authors  = "Gabriel Kreindler",
    modules  = [LogitTools],
    format   = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical  = "https://Gkreindler.github.io/LogitTools.jl",
    ),
    pages = [
        "Home"                 => "index.md",
        "Binary logit"         => "logit2.md",
        "Random coefficients"  => "logit2_rfx.md",
        "Option-level rfx"     => "mlogit_rfx.md",
        "API reference"        => "api.md",
    ],
    # Existing logit2/mlogit helpers are not all docstring'd yet; do not fail
    # the build over it.
    warnonly = [:missing_docs],
)

deploydocs(
    repo   = "github.com/Gkreindler/LogitTools.jl",
    devbranch = "main",
)
