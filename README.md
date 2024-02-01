# LogitTools

[![Build Status](https://github.com/Gkreindler/LogitTools.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Gkreindler/LogitTools.jl/actions/workflows/CI.yml?query=branch%3Amain)


Package to estimate binary logit models, Bayesian bootstrap (with option to cluster), and write to tables using RegressionTables.jl. Analytic likelihood and gradient + Optim.jl.

Modelled after [GMMTools.jl](https://github.com/Gkreindler/GMMTools.jl).

To install:
```
] add https://github.com/Gkreindler/LogitTools.jl
```

See examples/example.jl for basic usage.

Todo: 
- add other discrete choice models
- use actual formula from StatsAPI
- fixed effects
- code asymptotic variance covariance (?)