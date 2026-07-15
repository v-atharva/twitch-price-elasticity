# Cross-check: estimate the same CS spec as the Python pipeline using R's `did`
# (Callaway & Sant'Anna's reference implementation) on an exported panel CSV.
#
# usage: Rscript scripts/crosscheck.R <panel.csv> <out.csv> [anticipation]
# panel.csv columns: id (int), mindex (int), g (int, 0 = never treated),
#                    log_subs, pre_size, cat_share_games

args <- commandArgs(trailingOnly = TRUE)
panel_path <- args[[1]]
out_path <- args[[2]]
anticipation <- if (length(args) >= 3) as.integer(args[[3]]) else 0L

suppressMessages(library(did))

df <- read.csv(panel_path)

# mirror the Python wrapper: drop zero-variance covariates (singular DR design)
covs <- c("pre_size", "cat_share_games")
covs <- covs[sapply(covs, function(c) length(unique(df[[c]])) > 1)]
xformla <- if (length(covs)) as.formula(paste("~", paste(covs, collapse = " + "))) else ~1

res <- att_gt(
  yname = "log_subs",
  tname = "mindex",
  idname = "id",
  gname = "g",
  xformla = xformla,
  data = df,
  est_method = "dr",
  control_group = "nevertreated",
  anticipation = anticipation,
  base_period = "varying",
  allow_unbalanced_panel = TRUE,
  bstrap = FALSE,
  cband = FALSE
)

simple <- aggte(res, type = "simple", bstrap = FALSE, cband = FALSE, na.rm = TRUE)
dyn <- aggte(res, type = "dynamic", bstrap = FALSE, cband = FALSE, na.rm = TRUE)

out <- rbind(
  data.frame(kind = "simple", rel_period = NA, att = simple$overall.att, se = simple$overall.se),
  data.frame(kind = "event", rel_period = dyn$egt, att = dyn$att.egt, se = dyn$se.egt)
)
write.csv(out, out_path, row.names = FALSE)
cat(sprintf("R did: overall ATT %.6f (se %.6f), %d event-study points\n",
            simple$overall.att, simple$overall.se, length(dyn$egt)))
