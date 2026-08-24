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
theta0_rfx_multistart
boot_logit2_rfx
mlogit_rfx
theta0_mlogit_rfx
theta0_mlogit_rfx_multistart
boot_mlogit_rfx
rfx_term
rfx_cell_report
rfx_level_moments
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
LogitTools._rfx_fill_A!
LogitTools._logit2_rfx_multi
LogitTools._rfx_log_param_rows
LogitTools._rfx_log_param_stats
LogitTools._rfx_to_level
LogitTools._rfx_boot_se
LogitTools._boot_keep_rows
LogitTools._assemble_rfx_boot
LogitTools.MlogitGroups
LogitTools._make_cell_draws
LogitTools.MlogitRfxTerm
LogitTools.MlogitRfxPrep
LogitTools.MlogitRfxBuffers
LogitTools._normalize_mlogit_rfx
LogitTools._prep_mlogit_rfx
LogitTools._mlogit_rfx_cell_stats
LogitTools._mlogit_rfx_fg!
LogitTools._mlogit_rfx_ess
LogitTools._mlogit_rfx
LogitTools._mlogit_rfx_multi
LogitTools._mlogit_rfx_fill_A!
LogitTools._contiguous_blocks
```

## Index

```@index
```
