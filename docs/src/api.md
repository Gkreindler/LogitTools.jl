```@meta
CurrentModule = LogitTools
```

# API reference

## Binary logit

```@docs
logit2
boot_logit2
bbw!
```

## Multinomial logit

```@docs
mlogit
boot_mlogit
```

## Random-coefficient models

```@docs
logit2_rfx
theta0_rfx
boot_logit2_rfx
boot_report
boot_vcov!
regtable_rfx
```

## Internals

Not exported, but useful when extending the package.

```@docs
LogitTools._make_draws
LogitTools.RfxPrep
LogitTools.RfxBuffers
LogitTools._rfx_fg!
LogitTools._rfx_ess
LogitTools._logit2_rfx
LogitTools._prep_logit2_rfx
LogitTools._rfx_boot_weights
```

## Index

```@index
```
