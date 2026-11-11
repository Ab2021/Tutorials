---
title: "Feature engineering patterns"
topic: "Feature engineering patterns"
collection: "Part 03"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Feature engineering patterns

> **Evidence boundary:** All numbers and records below are ILLUSTRATIVE teaching inputs. Derivations are checked arithmetic, not measured deployment performance or a claim about any organisation.

## 1. File Header
**Purpose:** Represent decision-time evidence in a form a chosen learner can fit, serve, audit, and retire without changing its meaning.
**Decision contract:** For row i, emit a feature vector x_i using only facts available no later than the row cutoff c_i.
**Primary invariant:** For every component x_ij, `availability_time(x_ij) <= c_i`; a value failing this inequality is unavailable even when stored later.
**Population contract:** The training, validation, and scoring rows represent the same prediction moment, entity grain, and label horizon.
**Outcome:** Given raw fields, timestamps, model family, and cost limit, select a representation and write its availability proof.
**Prerequisites:** Train/validation separation, basic descriptive statistics, event time, processing time, and categorical data handling.
**Scope:** Tabular, event, text-derived, and entity-history features; optimisation of model hyperparameters is outside this sheet.
**Non-goal:** A correlation, importance score, or high offline metric does not by itself justify a feature.
**Notation:** i indexes rows, j indexes features, t is event time, c_i is decision cutoff, and y_i is an outcome observed after c_i.
**Evidence status:** DERIVED mechanisms and ILLUSTRATIVE arithmetic; no operational outcome is asserted.
**Audit record:** Store raw source, transform version, fit scope, cutoff rule, null rule, units, and owner for every feature.
**Release gate:** A feature is eligible only after train-only fitting, point-in-time replay, schema checks, and a cost estimate.
**Takeaway:** Feature engineering is constrained representation design, not a search for any variable correlated with a future label.

## 2. Concept Map / Mental Model
```text
raw event + event time + availability time
              |
              v
      cutoff-aware extraction ----> missingness policy
              |                         |
              v                         v
      fit transform on training    encoded feature vector
              |                         |
              +------ validation ------+
                        |
                        v
          serve the identical version at decision time
```
A raw field is an observation; a feature is a deterministic, versioned claim about how that observation may be used at a particular decision instant.
Time has three roles: event time says when a thing occurred, availability time says when it became usable, and processing time says when a pipeline handled it.
A feature can be causally irrelevant yet predictively useful, but it cannot rely on information unavailable to the actual decision.
Transform fitting is a learned operation when it estimates means, categories, bin edges, frequency counts, vocabulary, or target rates.
Fitting on validation rows leaks distributional information even when labels are never read; use training rows or fold-specific training partitions.
Feature selection is also fitting when the selection rule examines labels, so repeat it inside each validation fold.
A representation has an information benefit, a statistical risk, a latency cost, a storage cost, an operational dependency, and an interpretability cost.
A missing value is often a state of the measurement process; replacing it silently can destroy a useful availability signal.
Cyclical variables require topology-aware representation because Sunday and Monday are adjacent even though integer codes place them far apart.
One-hot encoding creates a separate axis for each category; hashing bounds width by accepting collisions; embedding learns a dense coordinate only with enough evidence.
Aggregations turn histories into snapshots; their window must end strictly before the prediction cutoff unless the decision is explicitly delayed.
Ratios express a relationship but introduce a denominator validity condition and a choice for zero, negative, or stale denominators.
Interactions add conditional structure but consume support; sparse crosses can memorise identifiers rather than generalise.
Scaling changes geometry for distance- and gradient-based learners, while tree split rules normally depend only on order and need not benefit from standardisation.
The correct baseline is usually the least transformed representation that meets availability and capacity constraints.
A feature specification behaves like an interface: changes in units, source semantics, timezone, or null coding can be breaking changes.
A feature store is not a proof of correctness; point-in-time joins and freshness monitoring still determine whether its values are admissible.
Offline parity means a replay with historical cutoffs yields the same transform logic as serving, not merely similar column names.
A model can tolerate noisy features better than a silent semantic swap because noise may average out while a swap changes the question being answered.
The design loop is: state decision → freeze cutoff → inventory sources → define transform → prove availability → fit safely → validate by time → estimate cost → monitor semantics.
**Takeaway:** The key mental model is a timestamped contract from raw observation to a reproducible decision-time vector.

## 3. Foundational Definitions
### 3.1 `age_days`
**Meaning:** `age_days` is elapsed time since verified account creation.
**Type and units:** Its declared representation is days; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `age_days`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `age_days`.
**Validity condition:** nonnegative duration.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `age_days` before trusting model behaviour.
### 3.2 `spend_30d`
**Meaning:** `spend_30d` is sum of eligible purchases in the preceding 30 days.
**Type and units:** Its declared representation is currency; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `spend_30d`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `spend_30d`.
**Validity condition:** closed trailing window.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `spend_30d` before trusting model behaviour.
### 3.3 `login_count_7d`
**Meaning:** `login_count_7d` is number of successful logins in the preceding 7 days.
**Type and units:** Its declared representation is count; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `login_count_7d`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `login_count_7d`.
**Validity condition:** integer at least zero.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `login_count_7d` before trusting model behaviour.
### 3.4 `days_since_login`
**Meaning:** `days_since_login` is elapsed time since the most recent successful login.
**Type and units:** Its declared representation is days; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `days_since_login`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `days_since_login`.
**Validity condition:** missing when no prior login.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `days_since_login` before trusting model behaviour.
### 3.5 `region`
**Meaning:** `region` is service region assigned before the decision.
**Type and units:** Its declared representation is category; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `region`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `region`.
**Validity condition:** stable taxonomy version.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `region` before trusting model behaviour.
### 3.6 `device_family`
**Meaning:** `device_family` is client hardware family reported at request.
**Type and units:** Its declared representation is category; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `device_family`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `device_family`.
**Validity condition:** known or other.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `device_family` before trusting model behaviour.
### 3.7 `amount_log`
**Meaning:** `amount_log` is log1p of a nonnegative transaction amount.
**Type and units:** Its declared representation is dimensionless; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `amount_log`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `amount_log`.
**Validity condition:** amount at least zero.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `amount_log` before trusting model behaviour.
### 3.8 `income_band`
**Meaning:** `income_band` is predeclared income interval rather than raw income.
**Type and units:** Its declared representation is ordered category; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `income_band`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `income_band`.
**Validity condition:** band definitions frozen.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `income_band` before trusting model behaviour.
### 3.9 `weekday_sin`
**Meaning:** `weekday_sin` is sine encoding of local decision weekday.
**Type and units:** Its declared representation is dimensionless; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `weekday_sin`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `weekday_sin`.
**Validity condition:** period exactly seven days.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `weekday_sin` before trusting model behaviour.
### 3.10 `weekday_cos`
**Meaning:** `weekday_cos` is cosine encoding of local decision weekday.
**Type and units:** Its declared representation is dimensionless; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `weekday_cos`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `weekday_cos`.
**Validity condition:** same timezone as sine.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `weekday_cos` before trusting model behaviour.
### 3.11 `address_missing`
**Meaning:** `address_missing` is indicator that address was absent at decision time.
**Type and units:** Its declared representation is binary; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `address_missing`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `address_missing`.
**Validity condition:** 1 means absent.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `address_missing` before trusting model behaviour.
### 3.12 `merchant_hash`
**Meaning:** `merchant_hash` is fixed-seed bucket of merchant identifier.
**Type and units:** Its declared representation is bucket id; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `merchant_hash`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `merchant_hash`.
**Validity condition:** seed and bucket count fixed.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `merchant_hash` before trusting model behaviour.
### 3.13 `category_frequency`
**Meaning:** `category_frequency` is training-window count for a category.
**Type and units:** Its declared representation is count; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `category_frequency`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `category_frequency`.
**Validity condition:** fit on training rows only.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `category_frequency` before trusting model behaviour.
### 3.14 `prior_rate_oof`
**Meaning:** `prior_rate_oof` is out-of-fold smoothed target rate for a category.
**Type and units:** Its declared representation is probability; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `prior_rate_oof`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `prior_rate_oof`.
**Validity condition:** fold-isolated label use.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `prior_rate_oof` before trusting model behaviour.
### 3.15 `amount_over_limit`
**Meaning:** `amount_over_limit` is amount divided by contemporaneous approved limit.
**Type and units:** Its declared representation is ratio; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `amount_over_limit`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `amount_over_limit`.
**Validity condition:** limit strictly positive.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `amount_over_limit` before trusting model behaviour.
### 3.16 `session_gap`
**Meaning:** `session_gap` is time between current and preceding session.
**Type and units:** Its declared representation is minutes; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `session_gap`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `session_gap`.
**Validity condition:** ordered events per entity.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `session_gap` before trusting model behaviour.
### 3.17 `rolling_mean_14d`
**Meaning:** `rolling_mean_14d` is mean eligible amount over prior 14 days.
**Type and units:** Its declared representation is currency; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `rolling_mean_14d`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `rolling_mean_14d`.
**Validity condition:** events strictly before cutoff.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `rolling_mean_14d` before trusting model behaviour.
### 3.18 `rolling_std_14d`
**Meaning:** `rolling_std_14d` is standard deviation of prior eligible amounts.
**Type and units:** Its declared representation is currency; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `rolling_std_14d`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `rolling_std_14d`.
**Validity condition:** at least two observations.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `rolling_std_14d` before trusting model behaviour.
### 3.19 `unique_merchants_30d`
**Meaning:** `unique_merchants_30d` is distinct merchant count in trailing 30 days.
**Type and units:** Its declared representation is count; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `unique_merchants_30d`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `unique_merchants_30d`.
**Validity condition:** deduplicate merchant keys.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `unique_merchants_30d` before trusting model behaviour.
### 3.20 `text_length`
**Meaning:** `text_length` is character count after stated normalisation.
**Type and units:** Its declared representation is characters; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `text_length`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `text_length`.
**Validity condition:** encoding defined.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `text_length` before trusting model behaviour.
### 3.21 `token_count`
**Meaning:** `token_count` is token count under a pinned tokenizer.
**Type and units:** Its declared representation is tokens; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `token_count`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `token_count`.
**Validity condition:** tokenizer version pinned.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `token_count` before trusting model behaviour.
### 3.22 `has_digit`
**Meaning:** `has_digit` is indicator that text contains an ASCII digit.
**Type and units:** Its declared representation is binary; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `has_digit`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `has_digit`.
**Validity condition:** input text available.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `has_digit` before trusting model behaviour.
### 3.23 `ratio_clicks_views`
**Meaning:** `ratio_clicks_views` is clicks divided by views plus an explicit zero-view rule.
**Type and units:** Its declared representation is ratio; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `ratio_clicks_views`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `ratio_clicks_views`.
**Validity condition:** same observation window.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `ratio_clicks_views` before trusting model behaviour.
### 3.24 `interaction_age_region`
**Meaning:** `interaction_age_region` is cross of age band and region.
**Type and units:** Its declared representation is category; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `interaction_age_region`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `interaction_age_region`.
**Validity condition:** sufficient support assessed.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `interaction_age_region` before trusting model behaviour.
### 3.25 `rank_within_entity`
**Meaning:** `rank_within_entity` is rank among events visible before cutoff.
**Type and units:** Its declared representation is ordinal; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `rank_within_entity`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `rank_within_entity`.
**Validity condition:** tie rule fixed.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `rank_within_entity` before trusting model behaviour.
### 3.26 `lag_1_value`
**Meaning:** `lag_1_value` is previous observed value for same entity.
**Type and units:** Its declared representation is native unit; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `lag_1_value`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `lag_1_value`.
**Validity condition:** strictly earlier event time.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `lag_1_value` before trusting model behaviour.
### 3.27 `trend_7d`
**Meaning:** `trend_7d` is slope through prior daily aggregates.
**Type and units:** Its declared representation is unit per day; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `trend_7d`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `trend_7d`.
**Validity condition:** at least two eligible days.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `trend_7d` before trusting model behaviour.
### 3.28 `holiday_flag`
**Meaning:** `holiday_flag` is calendar indicator from versioned calendar.
**Type and units:** Its declared representation is binary; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `holiday_flag`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `holiday_flag`.
**Validity condition:** calendar known beforehand.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `holiday_flag` before trusting model behaviour.
### 3.29 `distance_km`
**Meaning:** `distance_km` is great-circle distance from two consented coordinates.
**Type and units:** Its declared representation is kilometres; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `distance_km`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `distance_km`.
**Validity condition:** coordinates valid.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `distance_km` before trusting model behaviour.
### 3.30 `z_amount_entity`
**Meaning:** `z_amount_entity` is amount centered and scaled by prior entity history.
**Type and units:** Its declared representation is dimensionless; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `z_amount_entity`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `z_amount_entity`.
**Validity condition:** history count and epsilon stated.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `z_amount_entity` before trusting model behaviour.
### 3.31 `rare_category`
**Meaning:** `rare_category` is indicator that training frequency is below threshold.
**Type and units:** Its declared representation is binary; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `rare_category`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `rare_category`.
**Validity condition:** threshold fitted without labels.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `rare_category` before trusting model behaviour.
### 3.32 `missing_count`
**Meaning:** `missing_count` is number of absent fields in a declared schema.
**Type and units:** Its declared representation is count; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `missing_count`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `missing_count`.
**Validity condition:** schema version fixed.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `missing_count` before trusting model behaviour.
### 3.33 `source_age_minutes`
**Meaning:** `source_age_minutes` is processing time minus source event time.
**Type and units:** Its declared representation is minutes; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `source_age_minutes`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `source_age_minutes`.
**Validity condition:** clocks monitored.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `source_age_minutes` before trusting model behaviour.
### 3.34 `is_weekend`
**Meaning:** `is_weekend` is local calendar Saturday/Sunday flag.
**Type and units:** Its declared representation is binary; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `is_weekend`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `is_weekend`.
**Validity condition:** timezone assigned.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `is_weekend` before trusting model behaviour.
### 3.35 `quantile_amount`
**Meaning:** `quantile_amount` is training-fitted quantile bin for amount.
**Type and units:** Its declared representation is ordinal; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `quantile_amount`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `quantile_amount`.
**Validity condition:** bin edges frozen.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `quantile_amount` before trusting model behaviour.
### 3.36 `signed_log_balance`
**Meaning:** `signed_log_balance` is sign(x)*log1p(abs(x)) for signed balance.
**Type and units:** Its declared representation is dimensionless; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `signed_log_balance`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `signed_log_balance`.
**Validity condition:** finite balance.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `signed_log_balance` before trusting model behaviour.
### 3.37 `unit_price`
**Meaning:** `unit_price` is total price divided by units.
**Type and units:** Its declared representation is currency per unit; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `unit_price`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `unit_price`.
**Validity condition:** units greater than zero.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `unit_price` before trusting model behaviour.
### 3.38 `volume_per_day`
**Meaning:** `volume_per_day` is quantity divided by active exposure days.
**Type and units:** Its declared representation is quantity per day; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `volume_per_day`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `volume_per_day`.
**Validity condition:** exposure positive.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `volume_per_day` before trusting model behaviour.
### 3.39 `geo_cell`
**Meaning:** `geo_cell` is coarse spatial cell from coordinates.
**Type and units:** Its declared representation is category; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `geo_cell`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `geo_cell`.
**Validity condition:** privacy policy permits.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `geo_cell` before trusting model behaviour.
### 3.40 `language_group`
**Meaning:** `language_group` is language detector output collapsed to supported groups.
**Type and units:** Its declared representation is category; downstream code must not reinterpret it as a different unit or label. This declaration applies specifically to `language_group`.
**Availability rule:** Compute it only when its source facts were available by cutoff c_i, then persist the cutoff used for reconstruction. This is the availability proof for `language_group`.
**Validity condition:** abstention retained.
**Diagnostic:** Compare training and serving null rate, range, category support, and source-time lag for `language_group` before trusting model behaviour.
**Definition—raw feature:** A raw feature is directly supplied at or before cutoff; it can still be invalid because of stale source time or wrong entity join.
**Definition—derived feature:** A derived feature applies a deterministic expression to admissible raw values and must record its transform version.
**Definition—fitted transform:** A fitted transform estimates parameters from a reference set, such as a mean, vocabulary, bin edge, or frequency map.
**Definition—label-derived feature:** A feature using y requires fold isolation during development and an explicit delayed-label strategy in serving.
**Definition—leakage:** Leakage occurs when model input conveys outcome, future, cross-split, or target information unavailable under the deployed decision contract.
**Definition—support:** Support is the number and diversity of applicable training observations behind a learned category, aggregate, or interaction.
**Definition—freshness:** Freshness is c_i minus the source availability time, not c_i minus the event time; a recent event can arrive late.
**Definition—schema:** A schema fixes field names, types, null meanings, units, allowed categories, and semantic version.
**Takeaway:** Definitions make assumptions testable; vague feature names are not an audit trail.
## 4. Core Formulas with Full Derivation
This section treats a transform as valid only when its inputs, fitted parameters, and evaluation row obey the same decision-time contract.

### 4.1 Base case
**Result F-01 — standardisation is an affine coordinate change.**
**Validity before use:** A numeric feature has finite training values, a nonzero training standard deviation, and a model whose optimisation or distance geometry is scale-sensitive.
```text
FORMULA [F-01]: z_i = (x_i - mu_train) / sigma_train
WHERE: x_i is a scalar feature in its native unit; mu_train is the training-set mean in that unit;
       sigma_train is the positive training-set standard deviation in that unit; z_i is dimensionless.
CONSTRAINT: fit mu_train and sigma_train on training rows only; require sigma_train > 0; preserve the same parameters at validation and serving time.
NUMERICAL EXAMPLE: training values are 8, 10, and 12 units; mu_train=(8+10+12)/3=10 units;
                   population sigma_train=sqrt(((8-10)^2+0^2+(12-10)^2)/3)=sqrt(8/3)=1.63299 units;
                   for x_i=12 units, z_i=(12-10)/1.63299=1.22474.
INTERPRETATION: z_i measures displacement in training standard deviations, not a new physical measurement.
```
**Derivation:** Subtracting mu_train centres the coordinate at zero; division by sigma_train removes the original unit because units divided by identical units cancel.
**Independent check:** 1.22474 × 1.63299 + 10 = 12.00000 units after rounding, recovering the raw value.
**Limit:** If every training value is 10, sigma_train=0 and the coordinate is undefined; drop the constant or map it deliberately to zero.
**Validity condition:** Never refit mu_train or sigma_train on a scored batch, because that makes a single row depend on other scoring rows.

**Result F-02 — a trailing aggregate excludes the decision event.**
**Validity before use:** Events have entity keys and event times, the prediction row has cutoff c_i, and the business action occurs before any event at or after c_i is known.
```text
FORMULA [F-02]: a_i(W) = sum_{e: entity(e)=entity(i), c_i-W <= t_e < c_i} v_e
WHERE: W is a positive lookback duration; t_e is event time; v_e is an eligible event value in stated units; a_i(W) shares v_e units.
CONSTRAINT: include only events available by c_i as well as event-time eligible; the upper interval boundary is open.
NUMERICAL EXAMPLE: W=7 days; c_i=2026-04-08 09:00; eligible prior values are 12, 5, and 9 dollars;
                   a_i(7d)=12+5+9=26 dollars; an event of 30 dollars at 09:00 is excluded.
INTERPRETATION: The aggregate summarises known history rather than the outcome-period event.
```
**Derivation:** Indicator membership in a half-open interval selects each prior eligible event once; summing selected values preserves currency units.
**Independent check:** 12+5=17 and 17+9=26 dollars; including the cutoff event would instead produce 56 dollars and violate the stated boundary.
**Limit:** Event time alone is insufficient if the 5-dollar event arrived after c_i; availability time then excludes it despite an earlier timestamp.
**Validity condition:** Store both the window bounds and an availability predicate in the feature definition.

### 4.2 General case
**Result F-03 — smoothed category rate shrinks unstable evidence toward a global prior.**
**Validity before use:** A category’s historical labels are permitted for the training calculation, category outcomes share a defined label horizon, and the estimate is produced out of fold for training rows.
```text
FORMULA [F-03]: r_g = (s_g + alpha * p) / (n_g + alpha)
WHERE: s_g is positive-label count for category g; n_g is labelled count; p is reference positive rate; alpha is prior strength in pseudo-counts; r_g is a probability.
CONSTRAINT: 0<=s_g<=n_g, 0<=p<=1, alpha>0; each training row must use statistics excluding its own validation fold.
NUMERICAL EXAMPLE: s_g=3, n_g=8, p=0.20, alpha=12; r_g=(3+12*0.20)/(8+12)=(3+2.4)/20=0.27.
INTERPRETATION: Eight observations move the estimate from 0.20 toward 0.375 but do not let a small group dominate.
```
**Derivation:** Treat alpha*p as prior positive pseudo-counts and alpha as prior total pseudo-counts, then divide updated positives by updated total.
**Independent check:** 0.27×20=5.4 pseudo-positive counts, exactly 3 observed plus 2.4 prior counts.
**Sensitivity:** With alpha=0, r_g=3/8=0.375; with alpha=72, r_g=(3+14.4)/80=0.2175, closer to p.
**Limit:** If category prevalence changes after training, shrinkage does not correct the drift; monitor calibration by category and time.
**Validity condition:** The global p and each category count must be fit without labels from the row’s held-out fold.

**Result F-04 — one-hot width follows observed vocabulary, not row count.**
**Validity before use:** A finite categorical vocabulary has a documented unknown policy and the downstream model accepts sparse or dense indicator columns.
```text
FORMULA [F-04]: x_{i,k} = 1(category_i=k), for k in K_train; width = |K_train| + u
WHERE: K_train is the retained training vocabulary; u is 1 when an explicit unknown bucket is reserved, otherwise 0; x_{i,k} is binary.
CONSTRAINT: categories absent from training must map to unknown or error; never add serving-only columns without a versioned model contract.
NUMERICAL EXAMPLE: K_train={bronze,silver,gold} and u=1; width=3+1=4; a silver row maps to [0,1,0,0].
INTERPRETATION: Equality is represented geometrically rather than as an arbitrary ordinal distance.
```
**Derivation:** Each retained category receives one coordinate; the indicator definition selects exactly one coordinate for a known single-valued category.
**Independent check:** The silver vector has coordinate sum 1; a missing or unseen value uses [0,0,0,1] under the stated policy.
**Limit:** A multi-valued field needs multi-hot semantics, where the coordinate sum can exceed one and cardinality changes cost.
**Validity condition:** Freeze K_train and column order with the model artifact.

**Result F-05 — hashing bounds encoded width but permits collisions.**
**Validity before use:** Exact category identity is unnecessary, collisions are acceptable for the task, and a stable hash seed plus bucket count can be served identically.
```text
FORMULA [F-05]: b(g) = h_seed(g) mod m; expected pair collisions approximately q(q-1)/(2m)
WHERE: g is a category token; h_seed is a deterministic hash; m is bucket count; q is distinct active categories.
CONSTRAINT: this expectation assumes near-uniform independent bucket assignment and counts colliding pairs, not necessarily collided tokens.
NUMERICAL EXAMPLE: q=1,000 and m=100,000; expected pairs=1000*999/(2*100000)=4.995.
INTERPRETATION: Fixed width trades interpretability and collision-free identity for bounded memory.
```
**Derivation:** There are q(q-1)/2 unordered category pairs, and each pair lands together with approximate probability 1/m.
**Independent check:** 1,000×999=999,000; 999,000/200,000=4.995 expected colliding pairs.
**Limit:** Adversarial or correlated tokens can violate uniform hashing assumptions, and collisions can hide an important rare category.
**Validity condition:** Pin seed, normalisation, signed-hash policy, and m in both training and serving.

### 4.3 Variants
**Result F-06 — cyclical encoding embeds a period on the unit circle.**
**Validity before use:** A variable has known period P>0 and equal endpoints should be nearby, such as hour of day or day of week.
```text
FORMULA [F-06]: s=sin(2*pi*u/P), q=cos(2*pi*u/P)
WHERE: u is phase in the same unit as P; s and q are dimensionless coordinates; pi is the circle constant.
CONSTRAINT: use a consistent origin and timezone; this representation assumes smooth cyclic proximity, not arbitrary holiday effects.
NUMERICAL EXAMPLE: P=24 hours; u=0 gives (0,1); u=23 gives (sin(23*pi/12),cos(23*pi/12)) approximately (-0.2588,0.9659).
INTERPRETATION: Hours 23 and 0 are close because their Euclidean distance is about 0.261 rather than 23 integer units.
```
**Derivation:** Mapping phase to angle 2*pi*u/P places each period position on a radius-one circle, so P apart returns to the same coordinate.
**Independent check:** For u=0, sin(0)=0 and cos(0)=1; for u=P, sin(2*pi)=0 and cos(2*pi)=1.
**Limit:** A periodic encoding cannot represent a discontinuous policy change at midnight without an additional feature.
**Validity condition:** Confirm the local calendar convention before deriving u.

**Result F-07 — a guarded ratio separates zero denominator from valid magnitude.**
**Validity before use:** Numerator and denominator measure compatible windows, denominator semantics are known, and zero is not silently converted into a large finite ratio.
```text
FORMULA [F-07]: r_i = a_i/b_i when b_i>0; z_i=1(b_i=0)
WHERE: a_i and b_i are nonnegative quantities over the same window; r_i is a ratio; z_i is a binary denominator-state indicator.
CONSTRAINT: choose a documented value or null for r_i when b_i=0; do not use an arbitrary epsilon without analysing its scale effect.
NUMERICAL EXAMPLE: a_i=18 clicks and b_i=240 views; r_i=18/240=0.075; z_i=0.
INTERPRETATION: The rate is 7.5 percent, while the separate indicator preserves the zero-exposure state.
```
**Derivation:** Division normalises the numerator by exposure; the indicator carries information lost when zero prevents normalisation.
**Independent check:** 0.075×240=18 clicks; if b_i=0, r_i is not mathematically defined and z_i=1.
**Limit:** A denominator can be positive but stale, so zero guarding does not prove same-window alignment.
**Validity condition:** Record the denominator window, update lag, and zero policy.

### 4.4 Limits and counterexamples
**Counterexample C-01:** Normalising each scoring batch by its own mean creates batch-dependent predictions; the same applicant can receive a different feature solely because another applicant entered the batch.
**Counterexample C-02:** Using a full-history account total for a decision on 1 June leaks purchases that occurred on 2 June, even if the table is queried on 3 June.
**Counterexample C-03:** Encoding an identifier as an integer makes category 900 appear farther from category 2 than category 3; that distance has no category semantics.
**Counterexample C-04:** Target encoding every row with its own label makes a singleton positive category equal one and can create near-perfect training fit with no deployable signal.
**Counterexample C-05:** Log transformation is not defined for negative values; adding one only handles zero after a nonnegative-domain proof.
**Counterexample C-06:** A rolling standard deviation from one prior event is underdetermined under the sample definition, so imputation needs an explicit low-history state.
**Counterexample C-07:** A distance from latitude/longitude is invalid across swapped coordinates or missing coordinate reference systems; geometry cannot repair bad location semantics.
**Counterexample C-08:** A weekday sine/cosine pair does not make a monthly salary cycle smooth when the underlying process follows payment dates rather than seven-day periodicity.
**Takeaway:** Every formula is a conditional tool: its scope, fitted state, and failure boundary belong in the feature specification.

## 5. Worked Numerical Example
**Scenario:** ILLUSTRATIVE binary-risk scoring at 2026-04-08 09:00 UTC for one account, using only facts available by that instant.
**Raw history:** Prior eligible purchase amounts within the trailing seven-day half-open window are 12, 5, and 9 currency units; the event at the cutoff is excluded.
**Derived aggregate:** By F-02, spend_7d=12+5+9=26 currency units.
**Current quantity:** The current request has 18 clicks and 240 views accumulated in the same closed window before cutoff.
**Derived ratio:** By F-07, clicks_per_view=18/240=0.075 and zero_views=0.
**Training reference:** Historical training spend values are 8, 10, and 12 currency units with population mean 10 and population standard deviation sqrt(8/3)=1.63299.
**Scaled value:** For current spend 26, z_spend=(26-10)/1.63299=9.79796; this extreme value prompts an outlier and source-semantic check rather than a claim of risk.
**Category evidence:** A category has 3 positive outcomes in 8 labelled historical rows; the fold-safe global rate is 0.20 and prior strength is 12.
**Smoothed rate:** By F-03, (3+12×0.20)/(8+12)=5.4/20=0.27.
**Calendar encoding:** At hour 23 in a 24-hour local clock, sine is approximately -0.2588 and cosine is approximately 0.9659.
**Independent arithmetic re-check:** 26-10=16; 16/1.63299=9.79796 after rounding; 0.075×240=18; 0.27×20=5.4.
**Sensitivity A:** If one late 9-unit event was unavailable at cutoff, spend_7d becomes 17 and z_spend=(17-10)/1.63299=4.28661.
**Sensitivity B:** If alpha rises from 12 to 72, the category rate becomes (3+72×0.20)/(8+72)=17.4/80=0.2175.
**Sensitivity C:** If views are zero, the rate is undefined; retain zero_views=1 and use the documented null or model-imputation branch rather than outputting infinity.
**Decision implication:** The arithmetic illustrates representation construction only; model coefficients, thresholds, and outcomes are intentionally not asserted.
**Takeaway:** Independent checks expose where a feature changes because facts changed, because fitting changed, or because the definition was violated.

### 4.3a Transformation implementation notes
**Variant V-01 — `log1p`:** Apply log1p only after proving x is nonnegative; its inverse is exp(z)-1, so document whether values are native or transformed.
**Boundary V-01:** The transform is invalid until its fit scope, null rule, units, and decision-time availability are written in the registry.
**Check V-01:** Replay one historical row with a frozen cutoff and compare raw input, parameters, output, and fallback branch.
**Variant V-02 — `winsorisation`:** Fit lower and upper clipping bounds on training values only, then count values clipped by side in every later population.
**Boundary V-02:** The transform is invalid until its fit scope, null rule, units, and decision-time availability are written in the registry.
**Check V-02:** Replay one historical row with a frozen cutoff and compare raw input, parameters, output, and fallback branch.
**Variant V-03 — `quantile binning`:** Learn bin edges from the training reference distribution; preserve an explicit underflow and overflow rule for future values.
**Boundary V-03:** The transform is invalid until its fit scope, null rule, units, and decision-time availability are written in the registry.
**Check V-03:** Replay one historical row with a frozen cutoff and compare raw input, parameters, output, and fallback branch.
**Variant V-04 — `robust scaling`:** Use training median and interquartile range when extreme values would dominate mean and standard deviation, while treating zero IQR separately.
**Boundary V-04:** The transform is invalid until its fit scope, null rule, units, and decision-time availability are written in the registry.
**Check V-04:** Replay one historical row with a frozen cutoff and compare raw input, parameters, output, and fallback branch.
**Variant V-05 — `rank transform`:** Use training empirical ranks for ordinal positioning, but recognise that ties, extrapolation, and new extrema require a serving policy.
**Boundary V-05:** The transform is invalid until its fit scope, null rule, units, and decision-time availability are written in the registry.
**Check V-05:** Replay one historical row with a frozen cutoff and compare raw input, parameters, output, and fallback branch.
**Variant V-06 — `min-max scaling`:** Fit training minimum and maximum, but do not assume serving values remain in [0,1] because an out-of-range value can be legitimate drift.
**Boundary V-06:** The transform is invalid until its fit scope, null rule, units, and decision-time availability are written in the registry.
**Check V-06:** Replay one historical row with a frozen cutoff and compare raw input, parameters, output, and fallback branch.
**Variant V-07 — `ordinal encoding`:** Use numeric codes only when the declared category order has stable monotone meaning and equal code spacing is not claimed.
**Boundary V-07:** The transform is invalid until its fit scope, null rule, units, and decision-time availability are written in the registry.
**Check V-07:** Replay one historical row with a frozen cutoff and compare raw input, parameters, output, and fallback branch.
**Variant V-08 — `binary encoding`:** Reserve binary coding for a true two-state proposition and distinguish unknown from false whenever observation can be absent.
**Boundary V-08:** The transform is invalid until its fit scope, null rule, units, and decision-time availability are written in the registry.
**Check V-08:** Replay one historical row with a frozen cutoff and compare raw input, parameters, output, and fallback branch.
**Variant V-09 — `frequency encoding`:** Count category frequency in the fitting partition, then route unseen categories to a stated default rather than a calculated global lookup.
**Boundary V-09:** The transform is invalid until its fit scope, null rule, units, and decision-time availability are written in the registry.
**Check V-09:** Replay one historical row with a frozen cutoff and compare raw input, parameters, output, and fallback branch.
**Variant V-10 — `count encoding`:** A count may encode exposure rather than preference; pair it with a time window and inspect whether different entities have different opportunity.
**Boundary V-10:** The transform is invalid until its fit scope, null rule, units, and decision-time availability are written in the registry.
**Check V-10:** Replay one historical row with a frozen cutoff and compare raw input, parameters, output, and fallback branch.
**Variant V-11 — `rare grouping`:** Choose a support threshold before validation and map tail categories to other, but retain a report of which original categories were grouped.
**Boundary V-11:** The transform is invalid until its fit scope, null rule, units, and decision-time availability are written in the registry.
**Check V-11:** Replay one historical row with a frozen cutoff and compare raw input, parameters, output, and fallback branch.
**Variant V-12 — `feature crossing`:** Create a cross only after checking joint support because two well-supported marginals can form almost empty cells.
**Boundary V-12:** The transform is invalid until its fit scope, null rule, units, and decision-time availability are written in the registry.
**Check V-12:** Replay one historical row with a frozen cutoff and compare raw input, parameters, output, and fallback branch.
**Variant V-13 — `polynomial expansion`:** Square or multiply variables only when the scale and missing-value policy make the resulting units and magnitude interpretable.
**Boundary V-13:** The transform is invalid until its fit scope, null rule, units, and decision-time availability are written in the registry.
**Check V-13:** Replay one historical row with a frozen cutoff and compare raw input, parameters, output, and fallback branch.
**Variant V-14 — `lag feature`:** Sort by entity and event time, resolve ties deterministically, and use availability time to reject late-arriving earlier events.
**Boundary V-14:** The transform is invalid until its fit scope, null rule, units, and decision-time availability are written in the registry.
**Check V-14:** Replay one historical row with a frozen cutoff and compare raw input, parameters, output, and fallback branch.
**Variant V-15 — `rolling count`:** Specify whether endpoints are open or closed because inclusion of the decision event changes both availability and interpretation.
**Boundary V-15:** The transform is invalid until its fit scope, null rule, units, and decision-time availability are written in the registry.
**Check V-15:** Replay one historical row with a frozen cutoff and compare raw input, parameters, output, and fallback branch.
**Variant V-16 — `exponentially weighted mean`:** Set decay in time units, not row counts, when event spacing varies; otherwise inactive entities receive accidental weighting.
**Boundary V-16:** The transform is invalid until its fit scope, null rule, units, and decision-time availability are written in the registry.
**Check V-16:** Replay one historical row with a frozen cutoff and compare raw input, parameters, output, and fallback branch.
**Variant V-17 — `text normalisation`:** Pin Unicode normalisation, case policy, tokenisation, and redaction; a normaliser change can alter vocabulary and length statistics.
**Boundary V-17:** The transform is invalid until its fit scope, null rule, units, and decision-time availability are written in the registry.
**Check V-17:** Replay one historical row with a frozen cutoff and compare raw input, parameters, output, and fallback branch.
**Variant V-18 — `date decomposition`:** Derive calendar pieces in the business timezone and retain the original timestamp for replay when daylight-saving transitions occur.
**Boundary V-18:** The transform is invalid until its fit scope, null rule, units, and decision-time availability are written in the registry.
**Check V-18:** Replay one historical row with a frozen cutoff and compare raw input, parameters, output, and fallback branch.
**Variant V-19 — `geospatial distance`:** State coordinate reference system and missing-coordinate policy; kilometres from longitude degrees are not a valid Euclidean calculation.
**Boundary V-19:** The transform is invalid until its fit scope, null rule, units, and decision-time availability are written in the registry.
**Check V-19:** Replay one historical row with a frozen cutoff and compare raw input, parameters, output, and fallback branch.
**Variant V-20 — `entity normalisation`:** Compute per-entity baseline from history before cutoff and add history_count so the model can distinguish a reliable baseline from a fallback.
**Boundary V-20:** The transform is invalid until its fit scope, null rule, units, and decision-time availability are written in the registry.
**Check V-20:** Replay one historical row with a frozen cutoff and compare raw input, parameters, output, and fallback branch.
**Variant V-21 — `source freshness`:** Represent source lag separately from the source value because a stale but plausible value should not be indistinguishable from a current one.
**Boundary V-21:** The transform is invalid until its fit scope, null rule, units, and decision-time availability are written in the registry.
**Check V-21:** Replay one historical row with a frozen cutoff and compare raw input, parameters, output, and fallback branch.
**Variant V-22 — `missing indicator`:** Add an indicator when absence can arise from a meaningful collection path; do not add it blindly if missingness is impossible by schema.
**Boundary V-22:** The transform is invalid until its fit scope, null rule, units, and decision-time availability are written in the registry.
**Check V-22:** Replay one historical row with a frozen cutoff and compare raw input, parameters, output, and fallback branch.
**Variant V-23 — `outlier flag`:** Define an outlier relative to a fixed training or entity-history reference, then review whether a data-unit shift explains the flag.
**Boundary V-23:** The transform is invalid until its fit scope, null rule, units, and decision-time availability are written in the registry.
**Check V-23:** Replay one historical row with a frozen cutoff and compare raw input, parameters, output, and fallback branch.
**Variant V-24 — `monotone bucket`:** Use bins to enforce a coarse monotone relation only after verifying enough support and avoiding bins learned with future outcome data.
**Boundary V-24:** The transform is invalid until its fit scope, null rule, units, and decision-time availability are written in the registry.
**Check V-24:** Replay one historical row with a frozen cutoff and compare raw input, parameters, output, and fallback branch.
**Variant V-25 — `hashing`:** Estimate collision exposure at the expected active vocabulary, then test whether collision-prone categories alter audit-critical decisions.
**Boundary V-25:** The transform is invalid until its fit scope, null rule, units, and decision-time availability are written in the registry.
**Check V-25:** Replay one historical row with a frozen cutoff and compare raw input, parameters, output, and fallback branch.
**Formula review protocol:** For each new equation, list its dimensional units, exact numerator, denominator, index population, fitted state, and material limit before running a model comparison.
**Formula review counterexample:** A numerically correct calculation remains unusable if a feature’s label horizon overlaps the action horizon.
**Takeaway:** Transform variety does not relax the availability proof; it makes the proof more important.

### 4.3b Representation governance
**Governance G-01:** A fitted transform artifact must carry a data snapshot identifier or training-window bounds, so later reconstruction does not silently select newer facts.
**Evidence G-01:** Record the observed branch counts and a point-in-time fixture that exercises this condition.
**Boundary G-01:** If the fixture cannot be reconstructed from retained source facts, the feature is not reproducibly defined.
**Governance G-02:** A deterministic transform still requires versioning when library behavior, timezone database, Unicode tables, or numerical precision can change its output.
**Evidence G-02:** Record the observed branch counts and a point-in-time fixture that exercises this condition.
**Boundary G-02:** If the fixture cannot be reconstructed from retained source facts, the feature is not reproducibly defined.
**Governance G-03:** A feature with a valid value but impossible unit should fail closed; a 5,000-percent rate and a 50-fold ratio are different encodings of the same quantity.
**Evidence G-03:** Record the observed branch counts and a point-in-time fixture that exercises this condition.
**Boundary G-03:** If the fixture cannot be reconstructed from retained source facts, the feature is not reproducibly defined.
**Governance G-04:** A null fallback should be chosen before model fitting because adding it after an incident creates an untested distributional branch.
**Evidence G-04:** Record the observed branch counts and a point-in-time fixture that exercises this condition.
**Boundary G-04:** If the fixture cannot be reconstructed from retained source facts, the feature is not reproducibly defined.
**Governance G-05:** A feature’s name should expose its window and semantics, such as spend_prior_7d, rather than an ambiguous label like recent_spend.
**Evidence G-05:** Record the observed branch counts and a point-in-time fixture that exercises this condition.
**Boundary G-05:** If the fixture cannot be reconstructed from retained source facts, the feature is not reproducibly defined.
**Governance G-06:** A high-cardinality category can be useful through aggregates, hashing, or grouping, but direct identity encoding needs a support and privacy review.
**Evidence G-06:** Record the observed branch counts and a point-in-time fixture that exercises this condition.
**Boundary G-06:** If the fixture cannot be reconstructed from retained source facts, the feature is not reproducibly defined.
**Governance G-07:** A feature can have low individual importance while enabling a useful interaction, so remove it only after measuring the selected model family under leakage-safe validation.
**Evidence G-07:** Record the observed branch counts and a point-in-time fixture that exercises this condition.
**Boundary G-07:** If the fixture cannot be reconstructed from retained source facts, the feature is not reproducibly defined.
**Governance G-08:** A feature drift dashboard should display raw source coverage and transform outputs because stable outputs can conceal compensating input shifts.
**Evidence G-08:** Record the observed branch counts and a point-in-time fixture that exercises this condition.
**Boundary G-08:** If the fixture cannot be reconstructed from retained source facts, the feature is not reproducibly defined.
**Governance G-09:** Training-serving skew includes different rounding, different null parsing, different category normalisation, and different cutoff inclusivity; each has a direct replay test.
**Evidence G-09:** Record the observed branch counts and a point-in-time fixture that exercises this condition.
**Boundary G-09:** If the fixture cannot be reconstructed from retained source facts, the feature is not reproducibly defined.
**Governance G-10:** Backfills should be marked separately from originally available records; using corrected history in offline training can make production freshness look better than it was.
**Evidence G-10:** Record the observed branch counts and a point-in-time fixture that exercises this condition.
**Boundary G-10:** If the fixture cannot be reconstructed from retained source facts, the feature is not reproducibly defined.
**Governance G-11:** A feature specification should name an accountable owner who approves semantic changes and a retirement condition when its source is discontinued.
**Evidence G-11:** Record the observed branch counts and a point-in-time fixture that exercises this condition.
**Boundary G-11:** If the fixture cannot be reconstructed from retained source facts, the feature is not reproducibly defined.
## 6. Visual / Geometric Intuition
A representation changes which rows appear near one another to a learner. Geometry is useful because it reveals accidental assumptions about distance, direction, and equality.

### 6.1 Numeric coordinates
Raw amounts place one unit of currency one coordinate unit apart; this is meaningful only when the learner can use scale without being dominated by magnitude.
Standardisation shifts the training centre to zero and makes one training standard deviation equal one coordinate unit; it does not remove skew or make values comparable in business meaning.
Log1p compresses large positive distances: moving from 1 to 2 changes log1p by more than moving from 1,001 to 1,002, reflecting multiplicative rather than additive sensitivity.
Clipping makes all values beyond a boundary coincide geometrically; it limits leverage but deliberately loses tail ordering.
Quantile bins turn a line into steps; rows inside a bin are equal to the encoded representation even if their raw values differ.
A missing indicator adds a perpendicular axis: two imputed-zero rows can be separated when one was observed as zero and the other was unobserved.

### 6.2 Category coordinates
One-hot vectors occupy orthogonal vertices: two different single-valued categories have squared Euclidean distance two, independent of arbitrary label order.
An unknown bucket places every unseen category at the same vertex, so the model cannot distinguish their identities but can recognise novelty.
Frequency encoding maps categories onto a single count line; unrelated categories with the same count become geometrically equal, which can be acceptable only if identity beyond popularity is unnecessary.
Target encoding maps categories near an estimated response rate; smoothing pulls low-support categories toward the global point and reduces their isolated leverage.
Hashing overlays multiple category vertices into one bucket coordinate; a collision is a forced geometric identification, not evidence that categories are alike.
Learned embeddings locate categories by task signal, but their axes have no inherent human semantics and their geometry depends on training objective and regularisation.

### 6.3 Time and sequences
A timestamp as an integer makes 23:00 and 00:00 far apart; sine and cosine wrap the clock around a circle so neighbouring phases remain close.
A lag feature is an arrow from a row to its preceding event; a missing arrow means no history, not necessarily a numeric zero.
A trailing window is a rectangle ending before cutoff on a timeline; moving the right edge to or past cutoff crosses the availability boundary.
An exponentially weighted aggregate assigns a decaying kernel to prior events, so recent events pull the summary more strongly than distant events in elapsed time.
A rolling mean can smooth episodic noise while hiding a sudden regime change; inspecting both level and window count makes that trade visible.

### 6.4 Interactions and ratios
A product interaction bends a linear decision surface in the original coordinates because its effect changes with the companion variable.
A category cross partitions a plane into cells; sparse cells have high variance because few rows anchor their estimated effect.
A ratio projects two measurements onto a relative scale, making proportional change visible while discarding absolute exposure unless the denominator is also retained.
Entity-normalised values compare a point to its own historical cloud; they can reveal a deviation while losing population-level magnitude if raw value is omitted.

### 6.5 Availability geometry
Imagine each source record as a point in a two-dimensional event-time versus availability-time plane.
The admissible region for row i lies below both event-time and availability-time cutoff rules stated by the feature contract.
Late events occupy an earlier event-time position but a later availability position, making them visually obvious leakage candidates.
A point-in-time join selects the latest permissible record in that region; selecting the latest recorded value regardless of availability crosses the boundary.
**Takeaway:** Geometric intuition clarifies what a learner can compare, but timestamps and source semantics decide whether a coordinate is legal.

## 7. Comparison / Selection Table
| Pattern | Choose when | Avoid when | Fit scope | Availability proof | Cost signal |
|---|---|---|---|---|---|
| **Raw numeric** | native units and monotone magnitude are meaningful | units mix or tails dominate optimisation | none | source value existed by cutoff | one scalar per row |
**Selection note—Raw numeric:** Prefer this representation only after comparing it to the simplest legal baseline on a cutoff-respecting validation split.
**Boundary—Raw numeric:** The selection changes if its stated availability proof or fit scope fails, even when offline accuracy rises.
| **Standard scaling** | linear, neural, distance, or regularised model uses numeric scale | tree splits are the only consumer and scale is already safe | training rows only | frozen mean and deviation are pre-cutoff artifacts | two parameters per column |
**Selection note—Standard scaling:** Prefer this representation only after comparing it to the simplest legal baseline on a cutoff-respecting validation split.
**Boundary—Standard scaling:** The selection changes if its stated availability proof or fit scope fails, even when offline accuracy rises.
| **Robust scaling** | heavy tails would distort mean and deviation | median or IQR are unstable because sample is tiny | training rows only | frozen quantiles are fitted before scoring | two quantiles per column |
**Selection note—Robust scaling:** Prefer this representation only after comparing it to the simplest legal baseline on a cutoff-respecting validation split.
**Boundary—Robust scaling:** The selection changes if its stated availability proof or fit scope fails, even when offline accuracy rises.
| **Log1p** | nonnegative values span orders of magnitude | negative values or additive interpretation is required | none after domain proof | raw value available by cutoff | one transcendental operation |
**Selection note—Log1p:** Prefer this representation only after comparing it to the simplest legal baseline on a cutoff-respecting validation split.
**Boundary—Log1p:** The selection changes if its stated availability proof or fit scope fails, even when offline accuracy rises.
| **Clipping** | extreme values are known artifacts or leverage is bounded by policy | tail magnitude is decision-critical | training or policy thresholds | threshold version precedes deployment | comparison plus branch |
**Selection note—Clipping:** Prefer this representation only after comparing it to the simplest legal baseline on a cutoff-respecting validation split.
**Boundary—Clipping:** The selection changes if its stated availability proof or fit scope fails, even when offline accuracy rises.
| **Quantile bins** | coarse nonlinear steps aid an interpretable model | smooth precision or extrapolation is required | training rows only | edges are frozen model state | binary width times bins |
**Selection note—Quantile bins:** Prefer this representation only after comparing it to the simplest legal baseline on a cutoff-respecting validation split.
**Boundary—Quantile bins:** The selection changes if its stated availability proof or fit scope fails, even when offline accuracy rises.
| **One-hot** | cardinality is moderate and category identity matters | vocabulary is huge or unseen rate is high | training vocabulary only | category normalised before cutoff | width proportional to categories |
**Selection note—One-hot:** Prefer this representation only after comparing it to the simplest legal baseline on a cutoff-respecting validation split.
**Boundary—One-hot:** The selection changes if its stated availability proof or fit scope fails, even when offline accuracy rises.
| **Ordinal code** | category order is genuine and documented | numeric spacing would be treated as distance | policy order only | order version available before score | one scalar per row |
**Selection note—Ordinal code:** Prefer this representation only after comparing it to the simplest legal baseline on a cutoff-respecting validation split.
**Boundary—Ordinal code:** The selection changes if its stated availability proof or fit scope fails, even when offline accuracy rises.
| **Frequency code** | prevalence is useful and identity need not remain separate | frequency changes rapidly or rare identity matters | training window only | count map excludes future rows | one lookup plus map |
**Selection note—Frequency code:** Prefer this representation only after comparing it to the simplest legal baseline on a cutoff-respecting validation split.
**Boundary—Frequency code:** The selection changes if its stated availability proof or fit scope fails, even when offline accuracy rises.
| **Target code** | category response signal is strong with enough support | out-of-fold isolation cannot be implemented | fold-specific training only | label maturity and fold timing proven | map plus delayed labels |
**Selection note—Target code:** Prefer this representation only after comparing it to the simplest legal baseline on a cutoff-respecting validation split.
**Boundary—Target code:** The selection changes if its stated availability proof or fit scope fails, even when offline accuracy rises.
| **Hashing** | bounded width is more important than reversibility | collision audit or strict identity attribution is needed | fixed seed and dimensions | same normaliser and hash deployed | fixed m buckets |
**Selection note—Hashing:** Prefer this representation only after comparing it to the simplest legal baseline on a cutoff-respecting validation split.
**Boundary—Hashing:** The selection changes if its stated availability proof or fit scope fails, even when offline accuracy rises.
| **Rare grouping** | tail categories lack support and other is meaningful | rare category is safety-critical or protected | training counts only | threshold and vocabulary frozen | reduced width |
**Selection note—Rare grouping:** Prefer this representation only after comparing it to the simplest legal baseline on a cutoff-respecting validation split.
**Boundary—Rare grouping:** The selection changes if its stated availability proof or fit scope fails, even when offline accuracy rises.
| **Cyclical sine/cosine** | periodic adjacency is meaningful | calendar effects are discontinuous and arbitrary | none after convention | timezone and phase known by cutoff | two trigonometric values |
**Selection note—Cyclical sine/cosine:** Prefer this representation only after comparing it to the simplest legal baseline on a cutoff-respecting validation split.
**Boundary—Cyclical sine/cosine:** The selection changes if its stated availability proof or fit scope fails, even when offline accuracy rises.
| **Lag** | previous state has temporal signal | event ordering or late arrivals are unresolved | none but sorted history | preceding record was available by cutoff | state lookup per entity |
**Selection note—Lag:** Prefer this representation only after comparing it to the simplest legal baseline on a cutoff-respecting validation split.
**Boundary—Lag:** The selection changes if its stated availability proof or fit scope fails, even when offline accuracy rises.
| **Rolling aggregate** | recent history should influence score | future corrections contaminate historical replay | window metadata only | half-open window and availability filter | stateful or range query |
**Selection note—Rolling aggregate:** Prefer this representation only after comparing it to the simplest legal baseline on a cutoff-respecting validation split.
**Boundary—Rolling aggregate:** The selection changes if its stated availability proof or fit scope fails, even when offline accuracy rises.
| **Entity z-score** | relative deviation matters across heterogeneous entities | history is short or baseline is unsafe | history before cutoff | baseline has adequate prior count | state per entity |
**Selection note—Entity z-score:** Prefer this representation only after comparing it to the simplest legal baseline on a cutoff-respecting validation split.
**Boundary—Entity z-score:** The selection changes if its stated availability proof or fit scope fails, even when offline accuracy rises.
| **Ratio plus zero flag** | relative rate and exposure state matter | denominator mismatches numerator window | none after policy | both operands are contemporaneous | division and indicator |
**Selection note—Ratio plus zero flag:** Prefer this representation only after comparing it to the simplest legal baseline on a cutoff-respecting validation split.
**Boundary—Ratio plus zero flag:** The selection changes if its stated availability proof or fit scope fails, even when offline accuracy rises.
| **Missing indicator** | absence reflects collection path or risk | missingness is impossible by construction | none | null observed at cutoff | one bit per field |
**Selection note—Missing indicator:** Prefer this representation only after comparing it to the simplest legal baseline on a cutoff-respecting validation split.
**Boundary—Missing indicator:** The selection changes if its stated availability proof or fit scope fails, even when offline accuracy rises.
| **Text length/token count** | message size or structure is useful | tokeniser or redaction differs by environment | pinned normaliser only | text available and permitted by cutoff | linear in text length |
**Selection note—Text length/token count:** Prefer this representation only after comparing it to the simplest legal baseline on a cutoff-respecting validation split.
**Boundary—Text length/token count:** The selection changes if its stated availability proof or fit scope fails, even when offline accuracy rises.
| **Geo cell** | coarse location has approved predictive use | precise location lacks consent or is unstable | fixed spatial scheme | coordinates and policy available | lookup or cell computation |
**Selection note—Geo cell:** Prefer this representation only after comparing it to the simplest legal baseline on a cutoff-respecting validation split.
**Boundary—Geo cell:** The selection changes if its stated availability proof or fit scope fails, even when offline accuracy rises.
| **Interaction** | domain mechanism predicts conditional effect | joint support is sparse | selection inside validation | both operands admissible | extra columns or model capacity |
**Selection note—Interaction:** Prefer this representation only after comparing it to the simplest legal baseline on a cutoff-respecting validation split.
**Boundary—Interaction:** The selection changes if its stated availability proof or fit scope fails, even when offline accuracy rises.
**Selection workflow:** First reject illegal sources, then select a representation with compatible geometry, then estimate support and operational cost, and only then compare validation metrics.
**Model-family boundary:** Trees can learn threshold interactions from unscaled numeric features, whereas linear models often need explicit nonlinear bases or interactions; neither exemption permits leakage.
**Privacy boundary:** A technically predictive representation can remain disallowed because consent, policy, fairness, or purpose limitation prohibits its use.
**Takeaway:** Selection is a constrained choice among representations, not a contest in which the highest offline score automatically wins.

## 8. Failure Modes & Edge Cases
### FE-01 future aggregate
**Symptom:** Offline metric jumps after adding lifetime spend.
**Competing causes:** The window included events after cutoff; a backfill supplied future corrections; or label horizon overlaps the aggregate.
**Discriminating check:** Replay several rows with source availability timestamps and inspect whether any selected event has availability_time greater than cutoff.
**Bounded correction:** Use a half-open window plus availability predicate, rebuild historical values, and verify every contributing event is admissible.
**Verification:** Re-run the named check on a point-in-time fixture and confirm the feature contract, not merely the metric, has changed as intended. Failure record: FE-01.
### FE-02 full-data target encoding
**Symptom:** Training performance is unusually high for singleton categories.
**Competing causes:** Row label leaked into its own encoding; folds were not isolated; or categories genuinely have stable large support.
**Discriminating check:** For each training row, recompute encoding while withholding its fold and compare to stored value; inspect singleton mapping distribution.
**Bounded correction:** Generate out-of-fold values, smooth by support, and verify a row never reads its own held-out label.
**Verification:** Re-run the named check on a point-in-time fixture and confirm the feature contract, not merely the metric, has changed as intended. Failure record: FE-02.
### FE-03 train-serve category drift
**Symptom:** Serving routes many values to unknown and score distribution shifts.
**Competing causes:** New real categories appeared; normalisation changed; upstream field semantics changed; or vocabulary artifact was not loaded.
**Discriminating check:** Compare raw token samples, normalised tokens, unknown rate, and model-artifact vocabulary version.
**Bounded correction:** Restore shared normaliser and artifact, choose an unknown policy, then verify replay outputs match serving for fixtures.
**Verification:** Re-run the named check on a point-in-time fixture and confirm the feature contract, not merely the metric, has changed as intended. Failure record: FE-03.
### FE-04 timezone mismatch
**Symptom:** Weekend and hour features disagree with business records around midnight.
**Competing causes:** UTC was used instead of local time; timezone label changed; or daylight-saving conversion was ambiguous.
**Discriminating check:** Test named timestamps on both daylight-saving transition boundaries and compare local calendar decomposition.
**Bounded correction:** Pin timezone source and ambiguity rule, regenerate features, and verify phase and weekday against fixture expectations.
**Verification:** Re-run the named check on a point-in-time fixture and confirm the feature contract, not merely the metric, has changed as intended. Failure record: FE-04.
### FE-05 unit scaling error
**Symptom:** A ratio or z-score produces orders-of-magnitude outliers.
**Competing causes:** Currency was read as cents; percent was read as fraction; denominator changed units; or a source emitted string locale formatting.
**Discriminating check:** Check dimensional metadata and quantiles against raw records, especially factor-of-100 and factor-of-1000 patterns.
**Bounded correction:** Enforce typed unit conversion at ingestion and verify round-trip examples in the feature registry.
**Verification:** Re-run the named check on a point-in-time fixture and confirm the feature contract, not merely the metric, has changed as intended. Failure record: FE-05.
### FE-06 stale source
**Symptom:** A feature is present but decisions degrade after upstream delay.
**Competing causes:** Source pipeline lag increased; event time was mistaken for availability time; or fallback cache is old.
**Discriminating check:** Plot c_i minus availability time separately from c_i minus event time and inspect freshness by source partition.
**Bounded correction:** Add freshness feature and age limit, route stale values to bounded fallback, and verify incident replay selects the fallback.
**Verification:** Re-run the named check on a point-in-time fixture and confirm the feature contract, not merely the metric, has changed as intended. Failure record: FE-06.
### FE-07 denominator collapse
**Symptom:** Conversion rate becomes infinity or enormous after traffic falls.
**Competing causes:** Views are zero; views were filtered differently than clicks; bot filtering changed; or window endpoints differ.
**Discriminating check:** Audit numerator and denominator event IDs, window bounds, and zero-denominator count for the same population.
**Bounded correction:** Use guarded ratio plus zero indicator and align filters/windows, then verify 0-view rows take the documented branch.
**Verification:** Re-run the named check on a point-in-time fixture and confirm the feature contract, not merely the metric, has changed as intended. Failure record: FE-07.
### FE-08 sparse interaction
**Symptom:** An interaction improves one validation split but reverses later.
**Competing causes:** Joint category cells are sparse; split grouping is wrong; or a segment changed.
**Discriminating check:** Tabulate joint support and fold-to-fold effect estimates; compare against a grouped or time-based split.
**Bounded correction:** Prune unsupported crosses, group categories, or regularise interaction terms; verify stability across legal folds.
**Verification:** Re-run the named check on a point-in-time fixture and confirm the feature contract, not merely the metric, has changed as intended. Failure record: FE-08.
### FE-09 entity join fan-out
**Symptom:** Rolling totals are implausibly large and row count grows after joining.
**Competing causes:** A nonunique dimension caused duplication; entity keys are mis-normalised; or an as-of join selected several records.
**Discriminating check:** Check join cardinality before and after, count contributing event IDs, and assert one feature row per prediction row.
**Bounded correction:** Enforce key uniqueness or deterministic tie-breaker, rebuild, and verify conservation of distinct contributing events.
**Verification:** Re-run the named check on a point-in-time fixture and confirm the feature contract, not merely the metric, has changed as intended. Failure record: FE-09.
### FE-10 batch-dependent scaling
**Symptom:** The same request receives different score inputs in different batches.
**Competing causes:** Scaler is fit per batch; imputer computes batch statistic; or serving artifact is missing.
**Discriminating check:** Score a fixed fixture alone and with unrelated rows, then compare transformed features byte-for-byte.
**Bounded correction:** Load frozen training parameters and verify fixture invariance across batch sizes.
**Verification:** Re-run the named check on a point-in-time fixture and confirm the feature contract, not merely the metric, has changed as intended. Failure record: FE-10.
### FE-11 missingness semantic swap
**Symptom:** Missing indicator changes abruptly while raw source coverage appears stable.
**Competing causes:** Null became empty string; default zero replaced null; schema parser changed; or upstream stopped collecting field.
**Discriminating check:** Compare raw payload state, parsed state, and indicator by schema version; sample records around change time.
**Bounded correction:** Version parsing rules, retain absence distinctions, and verify migration fixtures for null, empty, zero, and invalid cases.
**Verification:** Re-run the named check on a point-in-time fixture and confirm the feature contract, not merely the metric, has changed as intended. Failure record: FE-11.
### FE-12 hash collision harm
**Symptom:** A rare category appears to inherit behaviour from an unrelated common category.
**Competing causes:** True hash collision occurred; token normalisation collapsed strings; or data label drift is concentrated in the category.
**Discriminating check:** Compute bucket membership under pinned hash and inspect whether several high-impact tokens share a bucket.
**Bounded correction:** Increase bucket count, change to auditable grouping, or reserve critical categories; verify collision report and decision slices.
**Verification:** Re-run the named check on a point-in-time fixture and confirm the feature contract, not merely the metric, has changed as intended. Failure record: FE-12.
### FE-13 late event rewrite
**Symptom:** Historical replay cannot reproduce online feature values.
**Competing causes:** Backfilled events were included; availability metadata was discarded; or source correction overwrote original fact.
**Discriminating check:** Compare original availability snapshots to current table and quantify records added after each historical cutoff.
**Bounded correction:** Retain bitemporal source history or snapshot IDs, use as-of availability joins, and verify replay against captured online fixtures.
**Verification:** Re-run the named check on a point-in-time fixture and confirm the feature contract, not merely the metric, has changed as intended. Failure record: FE-13.
### FE-14 protected proxy
**Symptom:** A feature shows strong lift but raises governance concern.
**Competing causes:** Legitimate operational factor; proxy for protected attribute; geographic policy issue; or sampling artifact.
**Discriminating check:** Review feature lineage and subgroup error slices under approved governance procedure, not only global importance.
**Bounded correction:** Remove or constrain disallowed use, document review, and verify the replacement meets both policy and performance bounds.
**Verification:** Re-run the named check on a point-in-time fixture and confirm the feature contract, not merely the metric, has changed as intended. Failure record: FE-14.
### FE-15 numerical overflow
**Symptom:** Log, exponent, or multiplication generates inf or NaN in a small fraction of rows.
**Competing causes:** Unexpected domain value; float precision issue; malformed input; or incorrect transform order.
**Discriminating check:** Assert finite outputs by transform stage and retain examples with raw values and branch metadata.
**Bounded correction:** Guard domains, clip only with policy justification, and verify boundary fixtures at minimum and maximum valid values.
**Verification:** Re-run the named check on a point-in-time fixture and confirm the feature contract, not merely the metric, has changed as intended. Failure record: FE-15.
**Takeaway:** A production failure is diagnosable only when the feature definition names competing causes and a check capable of separating them.
### FE-16 constant feature
**Symptom:** A scaled column is all zeros or NaNs.
**Competing causes:** The raw field is truly constant; filtering removed variation; unit conversion rounded values; or a source is stuck.
**Discriminating check:** Measure distinct raw values before scaling, inspect source partitions, and compare against previous accepted range.
**Bounded correction:** Drop deliberate constants or repair the source; assert positive variance before applying a variance-based scaler.
**Verification:** Demonstrate the correction on retained fixture data and observe the relevant branch count returns within its declared bound. Failure record: FE-16.
### FE-17 order instability
**Symptom:** Lag values change when equal-time events are replayed.
**Competing causes:** Event identifiers lack a tie-breaker; ingestion order leaked into sorting; or duplicate events exist.
**Discriminating check:** Reorder equal-time events in a fixture and compare lag outputs and aggregate membership.
**Bounded correction:** Specify stable event-id tie-breaker and deduplication policy, then verify permutation invariance.
**Verification:** Demonstrate the correction on retained fixture data and observe the relevant branch count returns within its declared bound. Failure record: FE-17.
### FE-18 vocabulary explosion
**Symptom:** One-hot matrix memory grows unexpectedly after a source update.
**Competing causes:** Identifier was mistaken for a category; free text entered categorical field; or normalisation stopped grouping variants.
**Discriminating check:** Track distinct count, top category share, and encoded width by version; sample new tail tokens.
**Bounded correction:** Apply schema validation and documented rare-grouping or hashing policy, then verify width against the capacity budget.
**Verification:** Demonstrate the correction on retained fixture data and observe the relevant branch count returns within its declared bound. Failure record: FE-18.
### FE-19 calendar backfill
**Symptom:** A holiday flag changes for old rows after a calendar update.
**Competing causes:** Calendar provider corrected history; local jurisdiction changed; or version was not retained.
**Discriminating check:** Replay fixed dates using old and new calendar versions and list changed flags.
**Bounded correction:** Version the calendar artifact and decide whether a model retrain is required; verify historical reproducibility.
**Verification:** Demonstrate the correction on retained fixture data and observe the relevant branch count returns within its declared bound. Failure record: FE-19.
### FE-20 misleading importance
**Symptom:** Permutation importance says a feature is useless although removing it degrades a subgroup.
**Competing causes:** Correlated substitute masks it; global aggregation hides a slice; or permutation created impossible combinations.
**Discriminating check:** Measure ablation and conditional diagnostics by segment while checking feature correlations and joint support.
**Bounded correction:** Use importance as a prompt for analysis, not deletion proof; verify selected removal across approved slices.
**Verification:** Demonstrate the correction on retained fixture data and observe the relevant branch count returns within its declared bound. Failure record: FE-20.
### FE-21 privacy retention
**Symptom:** A historical feature replay succeeds but retains source data beyond allowed duration.
**Competing causes:** Feature store copied raw identifiers; retention policy differs; or aggregate cannot be justified as de-identified.
**Discriminating check:** Trace raw lineage and retention tags through offline and online storage paths.
**Bounded correction:** Minimise or delete disallowed fields, enforce retention controls, and verify lineage report satisfies approved policy.
**Verification:** Demonstrate the correction on retained fixture data and observe the relevant branch count returns within its declared bound. Failure record: FE-21.
### FE-22 source outage fallback
**Symptom:** Feature values abruptly become defaults during an upstream outage.
**Competing causes:** True outage; authentication failure; schema rejection; or circuit breaker opened due to latency.
**Discriminating check:** Inspect source success metrics, fallback branch counts, and signed freshness values for the same interval.
**Bounded correction:** Use a tested fallback with explicit missingness state and verify decision behaviour under a synthetic outage fixture.
**Verification:** Demonstrate the correction on retained fixture data and observe the relevant branch count returns within its declared bound. Failure record: FE-22.
### FE-23 retrospective label drift
**Symptom:** Target-rate feature calibration decays after label-definition revision.
**Competing causes:** Outcome policy changed; delayed labels matured differently; or historical labels were relabelled.
**Discriminating check:** Compare label version, maturity distribution, and encoding prior across training and current cohorts.
**Bounded correction:** Version label semantics, retrain only on comparable labels, and verify new out-of-fold construction before release.
**Verification:** Demonstrate the correction on retained fixture data and observe the relevant branch count returns within its declared bound. Failure record: FE-23.
## 9. Complexity & Cost Analysis
Cost analysis distinguishes offline fitting, historical backfill, online feature retrieval, transform computation, model input width, and retained state. A cheap arithmetic expression can still be expensive when it requires an unbounded history lookup.

### 9.1 Per-row transform costs
For n scoring rows and d scalar arithmetic transforms, elementwise operations such as subtraction, division, indicators, and fixed clipping cost O(n d) time and O(n d) materialised output storage if stored densely.
One-hot encoding with d categorical fields and total retained widths w has O(n d) lookup time but O(n w) dense storage; sparse storage instead holds O(n d) nonzero entries for single-valued categories.
A direct hash encoding of q tokens is O(q) hashing work per row and O(m) model width, where m is fixed bucket count; model cost depends on m even though input token cardinality grows.
Sorting events by entity and time costs O(E log E) in a general batch of E events; pre-sorted partitioned streams can avoid repeated global sorting but require ordering guarantees.
A naive rolling-window aggregate scans prior history per row and can approach O(E W) for W inspected events; a deque or prefix-state approach often gives O(E) processing for additive windows after ordering.
Exact distinct counts over a trailing window require more state than sums because an item must be removed only after its final active occurrence expires.
Target encoding map construction costs O(n) to accumulate category counts and O(k) memory for k retained categories, excluding fold multiplication.
Out-of-fold target encoding across f folds repeats fitting work roughly f times over partitions, while each row receives an estimate from a model of its complement.
Text length is O(characters); token count is O(characters) to O(tokens) depending on tokenizer, and vocabulary-dependent encodings add artifact storage.
Distance calculations are O(1) per pair but coordinate validation, geocoding, consent enforcement, and cache misses can dominate engineering cost.

### 9.2 Worked capacity estimate
**Scenario:** ILLUSTRATIVE batch scoring processes 2,400,000 rows each day. Each row emits 180 numeric float32 features, 12 binary indicators packed as bytes for this estimate, and 4 categorical integer codes stored as int32.
```text
FORMULA [C-01]: numeric_bytes = rows * features * bytes_per_float
WHERE: rows=2,400,000; features=180; bytes_per_float=4 bytes for float32.
CONSTRAINT: payload only; excludes file blocks, indexes, compression, replication, metadata, and process memory.
NUMERICAL EXAMPLE: 2,400,000 * 180 * 4 = 1,728,000,000 bytes.
INTERPRETATION: numeric feature payload is about 1.609 GiB because 1 GiB=1,073,741,824 bytes.
```
```text
FORMULA [C-02]: categorical_bytes = rows * codes * bytes_per_code
WHERE: codes=4 integer codes per row; bytes_per_code=4 bytes for int32.
CONSTRAINT: assumes fixed-width codes and no dictionary payload in this row-level estimate.
NUMERICAL EXAMPLE: 2,400,000 * 4 * 4 = 38,400,000 bytes = 36.621 MiB.
INTERPRETATION: code columns are smaller than a wide dense one-hot representation for high-cardinality fields.
```
```text
FORMULA [C-03]: throughput_required = rows / allowed_seconds
WHERE: rows=2,400,000; allowed_seconds=20 minutes * 60 seconds/minute=1,200 seconds.
CONSTRAINT: average throughput; it excludes retries, skew, queueing, and startup overhead.
NUMERICAL EXAMPLE: 2,400,000 / 1,200 = 2,000 rows/second.
INTERPRETATION: capacity must exceed 2,000 rows/second after accounting for headroom and tail latency.
```
**Independent arithmetic check:** 180×4=720 numeric bytes per row; 720×2,400,000=1,728,000,000 bytes. Four codes use 16 bytes per row; 16×2,400,000=38,400,000 bytes.
**Indicator qualification:** Twelve binary values add 28,800,000 bytes if stored as one byte each: 2,400,000×12=28,800,000; bit packing would change this storage assumption.
**Total payload estimate:** 1,728,000,000+38,400,000+28,800,000=1,795,200,000 bytes, approximately 1.672 GiB before overhead.
**Sensitivity—one-hot width:** Replacing four codes with 4,000 dense float32 indicators adds 2,400,000×4,000×4=38,400,000,000 bytes, about 35.763 GiB, before other columns.
**Sensitivity—headroom:** If a service must sustain 1.5 times average throughput for bursts, target capacity is 2,000×1.5=3,000 rows/second, not an observed benchmark.
**Boundary:** Compression, sparse matrices, batch size, model architecture, cache hit rate, and serialization can materially change actual resource use; this is a payload estimate only.

### 9.3 Operational cost decisions
Prefer a precomputed daily aggregate only when its staleness is acceptable and the backfill can preserve historical availability.
Prefer an online rolling aggregate when decision value requires current events and the state store can meet the latency and consistency contract.
Budget feature width jointly with model memory; a representation that fits extraction memory can still make model inference too slow.
Measure p50 and tail latency separately because a feature source with acceptable mean latency can cause decision timeout under burst contention.
Partition state by entity key carefully; skewed entities can create hot partitions and violate otherwise sound average-capacity arithmetic.
Set a cardinality budget per field and an aggregate width budget per model artifact; alert before a new source consumes both.
Track cost per feature family so an expensive low-value source can be retired without confusing it with model-level resource growth.
**Takeaway:** Capacity is a feature-design constraint; widen a representation only when its validated decision value justifies its state, latency, and recovery cost.

## 10. Key Theorems / Results Reference
**Affine invariance of order:** If a>0, x and a*x+b have the same order. Consequence: rank-based or threshold-tree splits preserve ordering under positive affine rescaling, although missingness and numerical precision remain separate concerns.
**Monotone-transform caution:** A strictly monotone transform preserves order but not distance, linearity, variance, or calibration. Consequence: log transforms can change a linear model’s functional hypothesis class.
**Bias-variance trade-off in smoothing:** Adding prior pseudo-counts lowers variance of small-category estimates by introducing bias toward the reference rate. Consequence: alpha is a stability choice, not a universal constant.
**Curse of dimensionality:** As dimension grows, distances can concentrate and support per cell falls. Consequence: blindly adding crosses or one-hot columns can weaken nearest-neighbour and density-based reasoning.
**Data-processing perspective:** A deterministic feature cannot create information absent from its inputs, but it can make useful structure accessible to a restricted model family. Consequence: compare representation against model capacity and legal inputs.
**Sufficient-statistic boundary:** A summary is sufficient only relative to a probabilistic model and parameter of interest. Consequence: a rolling mean is not generally a lossless substitute for event history.
**Exchangeability warning:** Random row splits estimate a future deployment condition only when rows are exchangeable relative to the intended prediction task. Consequence: entities, time, and interventions often require grouped or temporal validation.
**No-free-lunch practical lesson:** No encoding dominates across all data-generating processes, costs, and governance constraints. Consequence: selection must state the decision and validation evidence.
**Simpson-style aggregation warning:** An apparent global relationship can reverse within segments. Consequence: inspect transformed feature behaviour by relevant cohorts before making a global interpretation.
**Measurement-error result:** Noise and missingness can attenuate or distort estimated relationships; a precise-looking derived feature can encode a poor upstream measurement. Consequence: monitor source quality as well as model metrics.
**Hashing result:** Pigeonhole principle guarantees some collision whenever more distinct tokens than buckets are mapped into fixed buckets. Consequence: collision risk is a design choice, not a surprise.
**Takeaway:** Theorems guide assumptions; they do not waive the empirical, temporal, and operational checks required by a feature contract.

## 11. Interview / Exam Questions
### Q1 — Define a legal feature.
**Answer:** A legal feature is computed from facts available by the row’s cutoff, using a versioned transform whose fitted parameters were learned only from permitted reference data.
**Reasoning:** It separates raw correlation from decision-time admissibility and forces time, source, and fit-scope reasoning.
**Follow-up:** State the event, availability, and processing timestamps for a late-arriving event; only availability by cutoff establishes legality.
**Red flag:** An answer that says only “test it” without naming cutoff, fit scope, competing cause, or quantitative check is incomplete. Assessment item: Q1.
**Scoring guide:** Full credit requires the stated conclusion and its validity condition; partial credit identifies the technique but omits the boundary. Assessment item: Q1.
**Checked basis:** The arithmetic or diagnostic follows the feature definitions and formulas established in this sheet. Assessment item: Q1.
### Q2 — Calculate a trailing sum.
**Answer:** With values 7, 11, and 4 inside a half-open window and a 20-unit event exactly at cutoff, the sum is 7+11+4=22 units; the cutoff event is excluded.
**Reasoning:** The open right endpoint prevents the current decision event from entering its own predictor.
**Follow-up:** If the 11-unit event was available after cutoff, the legal sum falls to 11 units despite its earlier event timestamp.
**Red flag:** An answer that says only “test it” without naming cutoff, fit scope, competing cause, or quantitative check is incomplete. Assessment item: Q2.
**Scoring guide:** Full credit requires the stated conclusion and its validity condition; partial credit identifies the technique but omits the boundary. Assessment item: Q2.
**Checked basis:** The arithmetic or diagnostic follows the feature definitions and formulas established in this sheet. Assessment item: Q2.
### Q3 — Diagnose a target-encoding leak.
**Answer:** A singleton category encoded as 1.0 for its only positive training row is suspicious; recompute its value from other folds and it should revert toward the prior when no other support exists.
**Reasoning:** Fold isolation prevents the row’s own label from creating its predictor.
**Follow-up:** A high rate can still be legitimate with many pre-cutoff labelled examples, so inspect support and fold lineage.
**Red flag:** An answer that says only “test it” without naming cutoff, fit scope, competing cause, or quantitative check is incomplete. Assessment item: Q3.
**Scoring guide:** Full credit requires the stated conclusion and its validity condition; partial credit identifies the technique but omits the boundary. Assessment item: Q3.
**Checked basis:** The arithmetic or diagnostic follows the feature definitions and formulas established in this sheet. Assessment item: Q3.
### Q4 — Choose encoding for an ordered band.
**Answer:** Use ordinal encoding only if the bands have a declared stable order; otherwise one-hot encoding avoids claiming that adjacent labels are equally spaced.
**Reasoning:** Integer codes create a numerical geometry that a linear learner may treat as distance.
**Follow-up:** If the model is a tree, ordinal codes can still create arbitrary split ordering, so compare against one-hot under legal validation.
**Red flag:** An answer that says only “test it” without naming cutoff, fit scope, competing cause, or quantitative check is incomplete. Assessment item: Q4.
**Scoring guide:** Full credit requires the stated conclusion and its validity condition; partial credit identifies the technique but omits the boundary. Assessment item: Q4.
**Checked basis:** The arithmetic or diagnostic follows the feature definitions and formulas established in this sheet. Assessment item: Q4.
### Q5 — Compute a smoothed rate.
**Answer:** For s=6, n=10, p=0.30, alpha=20, the rate is (6+20×0.30)/(10+20)=12/30=0.40.
**Reasoning:** Six observed positives combine with six prior pseudo-positives over thirty pseudo-observations.
**Follow-up:** If alpha becomes zero, the unsmoothed rate is 0.60; explain why the difference is larger for low-support categories.
**Red flag:** An answer that says only “test it” without naming cutoff, fit scope, competing cause, or quantitative check is incomplete. Assessment item: Q5.
**Scoring guide:** Full credit requires the stated conclusion and its validity condition; partial credit identifies the technique but omits the boundary. Assessment item: Q5.
**Checked basis:** The arithmetic or diagnostic follows the feature definitions and formulas established in this sheet. Assessment item: Q5.
### Q6 — Explain why batch scaling is unsafe.
**Answer:** A scaler fitted on each request batch changes a row’s transformed value when unrelated rows arrive, so the model input is not a function of the row and frozen artifact alone.
**Reasoning:** That violates reproducibility and can produce inconsistent decisions.
**Follow-up:** Demonstrate it by scoring one fixture alone and in a batch, then require exact equality of transformed outputs.
**Red flag:** An answer that says only “test it” without naming cutoff, fit scope, competing cause, or quantitative check is incomplete. Assessment item: Q6.
**Scoring guide:** Full credit requires the stated conclusion and its validity condition; partial credit identifies the technique but omits the boundary. Assessment item: Q6.
**Checked basis:** The arithmetic or diagnostic follows the feature definitions and formulas established in this sheet. Assessment item: Q6.
### Q7 — Select a cyclic representation.
**Answer:** For hour of day, use sine and cosine with period 24 when midnight adjacency matters; retain a timezone rule.
**Reasoning:** The pair maps the periodic boundary onto a circle rather than a broken integer line.
**Follow-up:** If effects are driven by scheduled shifts rather than smooth time, add shift indicators or another lawful representation.
**Red flag:** An answer that says only “test it” without naming cutoff, fit scope, competing cause, or quantitative check is incomplete. Assessment item: Q7.
**Scoring guide:** Full credit requires the stated conclusion and its validity condition; partial credit identifies the technique but omits the boundary. Assessment item: Q7.
**Checked basis:** The arithmetic or diagnostic follows the feature definitions and formulas established in this sheet. Assessment item: Q7.
### Q8 — Size a dense feature payload.
**Answer:** For 500,000 rows and 300 float32 features, bytes equal 500,000×300×4=600,000,000 bytes, approximately 572.205 MiB.
**Reasoning:** Each float32 consumes four bytes and 1 MiB is 1,048,576 bytes.
**Follow-up:** Add metadata, temporary copies, and model working memory before declaring a machine adequate.
**Red flag:** An answer that says only “test it” without naming cutoff, fit scope, competing cause, or quantitative check is incomplete. Assessment item: Q8.
**Scoring guide:** Full credit requires the stated conclusion and its validity condition; partial credit identifies the technique but omits the boundary. Assessment item: Q8.
**Checked basis:** The arithmetic or diagnostic follows the feature definitions and formulas established in this sheet. Assessment item: Q8.
### Q9 — Separate two causes of unknown categories.
**Answer:** Compare raw source token distribution to normalised tokens and loaded vocabulary version: novel raw tokens indicate real drift, while familiar raw tokens becoming unknown after normalisation indicate a transform mismatch.
**Reasoning:** Both causes yield unknowns, but their evidence appears at different stages of the pipeline.
**Follow-up:** A vocabulary artifact loading failure can mimic both, so include artifact checksum in the check.
**Red flag:** An answer that says only “test it” without naming cutoff, fit scope, competing cause, or quantitative check is incomplete. Assessment item: Q9.
**Scoring guide:** Full credit requires the stated conclusion and its validity condition; partial credit identifies the technique but omits the boundary. Assessment item: Q9.
**Checked basis:** The arithmetic or diagnostic follows the feature definitions and formulas established in this sheet. Assessment item: Q9.
### Q10 — Handle zero denominator.
**Answer:** Do not emit infinity; store a documented null or fallback value for the ratio and a separate zero-denominator indicator, after confirming numerator and denominator use the same window.
**Reasoning:** Zero exposure is a state that division cannot represent numerically.
**Follow-up:** If zero is semantically impossible, treat it as a data-quality failure rather than a normal model branch.
**Red flag:** An answer that says only “test it” without naming cutoff, fit scope, competing cause, or quantitative check is incomplete. Assessment item: Q10.
**Scoring guide:** Full credit requires the stated conclusion and its validity condition; partial credit identifies the technique but omits the boundary. Assessment item: Q10.
**Checked basis:** The arithmetic or diagnostic follows the feature definitions and formulas established in this sheet. Assessment item: Q10.
### Q11 — Assess an interaction proposal.
**Answer:** Before adding region×device, tabulate joint support by training fold, define an unknown/rare policy, and compare a regularised interaction against a main-effects baseline on time-respecting validation.
**Reasoning:** Strong marginal support does not guarantee sufficient support in each cross cell.
**Follow-up:** If the interaction helps only a random split, test whether entity or time leakage created the apparent lift.
**Red flag:** An answer that says only “test it” without naming cutoff, fit scope, competing cause, or quantitative check is incomplete. Assessment item: Q11.
**Scoring guide:** Full credit requires the stated conclusion and its validity condition; partial credit identifies the technique but omits the boundary. Assessment item: Q11.
**Checked basis:** The arithmetic or diagnostic follows the feature definitions and formulas established in this sheet. Assessment item: Q11.
### Q12 — Decide whether a feature is ready.
**Answer:** Require source lineage, cutoff rule, null semantics, units, transform artifact, train-serving fixture parity, validation result, cost estimate, and monitoring owner; lacking any one leaves an unbounded risk.
**Reasoning:** Readiness is an operational and semantic decision, not merely an offline feature-importance ranking.
**Follow-up:** For a delayed source, add a freshness policy and prove fallback behaviour before approval.
**Red flag:** An answer that says only “test it” without naming cutoff, fit scope, competing cause, or quantitative check is incomplete. Assessment item: Q12.
**Scoring guide:** Full credit requires the stated conclusion and its validity condition; partial credit identifies the technique but omits the boundary. Assessment item: Q12.
**Checked basis:** The arithmetic or diagnostic follows the feature definitions and formulas established in this sheet. Assessment item: Q12.
**Takeaway:** Strong answers name the representation, the decision-time constraint, and the observation that would falsify the proposed explanation.


### 9.4 Cost-control patterns
**Cost pattern P-01 — Materialise once:** Precompute a stable daily feature when many downstream decisions reuse it and the stated freshness budget permits reuse.
**Evidence P-01:** Compare stale-value age against decision tolerance, then measure recomputation cost and replay parity.
**Limit P-01:** A cost reduction is invalid if it changes the feature’s declared cutoff, precision, access control, or failure semantics.
**Cost pattern P-02 — Compute on demand:** Calculate a feature online when its latest signal changes decisions enough to justify lookup latency and failure handling.
**Evidence P-02:** Load-test dependency tails and inject an outage to observe the bounded fallback branch.
**Limit P-02:** A cost reduction is invalid if it changes the feature’s declared cutoff, precision, access control, or failure semantics.
**Cost pattern P-03 — Keep sparse:** Store wide one-hot or multi-hot inputs sparsely when nonzero count is far below total width.
**Evidence P-03:** Measure nonzero density and ensure the learner actually accepts the sparse format without hidden densification.
**Limit P-03:** A cost reduction is invalid if it changes the feature’s declared cutoff, precision, access control, or failure semantics.
**Cost pattern P-04 — Group tails:** Map low-support categories to other before matrix construction when exact identity adds width without validated stable signal.
**Evidence P-04:** Track grouped-category share, then inspect whether a critical category is being hidden.
**Limit P-04:** A cost reduction is invalid if it changes the feature’s declared cutoff, precision, access control, or failure semantics.
**Cost pattern P-05 — Bound history:** Retain only the window and sufficient state needed for legal aggregates rather than an unbounded event log in the hot path.
**Evidence P-05:** Verify expiry removes contributions at the exact window boundary using a time-travel fixture.
**Limit P-05:** A cost reduction is invalid if it changes the feature’s declared cutoff, precision, access control, or failure semantics.
**Cost pattern P-06 — Cache carefully:** Cache a source lookup only with explicit TTL, key scope, and invalidation semantics.
**Evidence P-06:** Compare cache age to source freshness limit and test entity-key isolation.
**Limit P-06:** A cost reduction is invalid if it changes the feature’s declared cutoff, precision, access control, or failure semantics.
**Cost pattern P-07 — Avoid duplicate extraction:** Share a canonical transform artifact across consumers only when their cutoff, purpose, and access policy are identical.
**Evidence P-07:** Compare lineage and authorisation contracts; reuse is unsafe when one consumer needs a different time boundary.
**Limit P-07:** A cost reduction is invalid if it changes the feature’s declared cutoff, precision, access control, or failure semantics.
**Cost pattern P-08 — Sample for exploration:** Use sampled data for prototype cost estimates but validate final definitions on full legal population before release.
**Evidence P-08:** Report sampling frame and compare category cardinality and tail rates to the full population.
**Limit P-08:** A cost reduction is invalid if it changes the feature’s declared cutoff, precision, access control, or failure semantics.
**Cost pattern P-09 — Version compactly:** Store transform configuration and checksums alongside model artifact rather than copying large mutable lookup maps without provenance.
**Evidence P-09:** Restore an archived artifact and reproduce a named historical fixture exactly.
**Limit P-09:** A cost reduction is invalid if it changes the feature’s declared cutoff, precision, access control, or failure semantics.
**Cost pattern P-10 — Retire actively:** Remove a feature after validated lack of value or source retirement to reduce cost, attack surface, and drift workload.
**Evidence P-10:** Ablate under current validation and confirm no dependent feature or policy still references it.
**Limit P-10:** A cost reduction is invalid if it changes the feature’s declared cutoff, precision, access control, or failure semantics.

### 10.1 Result application notes
**Result note R-01 — Order invariance:** Do not conclude that a tree needs no preprocessing: positive rescaling preserves ordering, but null encoding, category treatment, extreme precision, and illegal timestamps remain independent problems.
**Question R-01:** What would have to be true for this result not to apply to the proposed transform? Result record: R-01.
**Use R-01:** Write that answer into the feature’s validity condition before comparing model metrics.
**Result note R-02 — Smoothing:** Choose alpha through leakage-safe validation and support inspection; a selected alpha should be reported as a fitted modelling choice rather than an intrinsic property of a category.
**Question R-02:** What would have to be true for this result not to apply to the proposed transform? Result record: R-02.
**Use R-02:** Write that answer into the feature’s validity condition before comparing model metrics.
**Result note R-03 — Dimensionality:** Count both statistical and systems dimensions: each additional coordinate consumes support, artifact space, model capacity, and monitoring attention.
**Question R-03:** What would have to be true for this result not to apply to the proposed transform? Result record: R-03.
**Use R-03:** Write that answer into the feature’s validity condition before comparing model metrics.
**Result note R-04 — Determinism:** A deterministic transform can still be unreproducible if its lookup map, timezone library, random seed, or input ordering is not fixed.
**Question R-04:** What would have to be true for this result not to apply to the proposed transform? Result record: R-04.
**Use R-04:** Write that answer into the feature’s validity condition before comparing model metrics.
**Result note R-05 — Validation:** Temporal validation is a representation test because it exposes feature freshness, taxonomy drift, and future-information assumptions that random splits can hide.
**Question R-05:** What would have to be true for this result not to apply to the proposed transform? Result record: R-05.
**Use R-05:** Write that answer into the feature’s validity condition before comparing model metrics.
**Result note R-06 — Causality:** A predictive feature need not be causal, but causal plausibility can help identify unstable proxies and post-outcome variables.
**Question R-06:** What would have to be true for this result not to apply to the proposed transform? Result record: R-06.
**Use R-06:** Write that answer into the feature’s validity condition before comparing model metrics.
**Result note R-07 — Compression:** A low-dimensional encoding is lossy by design; evaluate whether the discarded distinctions are irrelevant to the permitted decision, not just to training loss.
**Question R-07:** What would have to be true for this result not to apply to the proposed transform? Result record: R-07.
**Use R-07:** Write that answer into the feature’s validity condition before comparing model metrics.
**Result note R-08 — Monitoring:** Drift statistics require a reference distribution with the same transform version; comparing raw and transformed values answers different questions.
**Question R-08:** What would have to be true for this result not to apply to the proposed transform? Result record: R-08.
**Use R-08:** Write that answer into the feature’s validity condition before comparing model metrics.
**Result note R-09 — Fairness:** A representation can create disparate performance through differential missingness, granularity, or proxy structure even when it excludes an explicitly protected field.
**Question R-09:** What would have to be true for this result not to apply to the proposed transform? Result record: R-09.
**Use R-09:** Write that answer into the feature’s validity condition before comparing model metrics.
**Result note R-10 — Reproducibility:** A reproduced number is necessary but not sufficient: confirm it came from the same legal historical inputs rather than corrected or future backfilled data.
**Question R-10:** What would have to be true for this result not to apply to the proposed transform? Result record: R-10.
**Use R-10:** Write that answer into the feature’s validity condition before comparing model metrics.


### 9.4 Cost-control patterns
**Cost pattern P-01 — Materialise once:** Precompute a stable daily feature when many decisions reuse it and the stated freshness budget permits reuse.
**Evidence P-01:** Compare stale-value age against decision tolerance, then measure recomputation cost and replay parity.
**Limit P-01:** A cost reduction is invalid if it changes declared cutoff, precision, access control, or failure semantics.
**Cost pattern P-02 — Compute on demand:** Calculate a feature online when its latest signal changes decisions enough to justify lookup latency and failure handling.
**Evidence P-02:** Load-test dependency tails and inject an outage to observe the bounded fallback branch.
**Limit P-02:** A cost reduction is invalid if it changes declared cutoff, precision, access control, or failure semantics.
**Cost pattern P-03 — Keep sparse:** Store wide one-hot or multi-hot inputs sparsely when nonzero count is far below total width.
**Evidence P-03:** Measure nonzero density and ensure the learner accepts sparse format without hidden densification.
**Limit P-03:** A cost reduction is invalid if it changes declared cutoff, precision, access control, or failure semantics.
**Cost pattern P-04 — Group tails:** Map low-support categories to other before matrix construction when exact identity adds width without stable signal.
**Evidence P-04:** Track grouped-category share, then inspect whether a critical category is being hidden.
**Limit P-04:** A cost reduction is invalid if it changes declared cutoff, precision, access control, or failure semantics.
**Cost pattern P-05 — Bound history:** Retain only the window and sufficient state needed for legal aggregates rather than an unbounded hot-path log.
**Evidence P-05:** Verify expiry removes contributions at the exact window boundary using a time-travel fixture.
**Limit P-05:** A cost reduction is invalid if it changes declared cutoff, precision, access control, or failure semantics.
**Cost pattern P-06 — Cache carefully:** Cache a source lookup only with explicit TTL, key scope, and invalidation semantics.
**Evidence P-06:** Compare cache age to source freshness limit and test entity-key isolation.
**Limit P-06:** A cost reduction is invalid if it changes declared cutoff, precision, access control, or failure semantics.
**Cost pattern P-07 — Avoid duplicate extraction:** Share a canonical transform artifact only when consumers have identical cutoff, purpose, and access policy.
**Evidence P-07:** Compare lineage and authorisation contracts before reusing an extraction.
**Limit P-07:** A cost reduction is invalid if it changes declared cutoff, precision, access control, or failure semantics.
**Cost pattern P-08 — Sample for exploration:** Use sampled data for prototype cost estimates but validate final definitions on the full legal population.
**Evidence P-08:** Report sampling frame and compare category cardinality and tail rates to the full population.
**Limit P-08:** A cost reduction is invalid if it changes declared cutoff, precision, access control, or failure semantics.
**Cost pattern P-09 — Version compactly:** Store transform configuration and checksums with the model artifact rather than unproven mutable lookup maps.
**Evidence P-09:** Restore an archived artifact and reproduce a named historical fixture exactly.
**Limit P-09:** A cost reduction is invalid if it changes declared cutoff, precision, access control, or failure semantics.
**Cost pattern P-10 — Retire actively:** Remove a feature after validated lack of value or source retirement to reduce cost and drift workload.
**Evidence P-10:** Ablate under current validation and confirm no dependent feature or policy still references it.
**Limit P-10:** A cost reduction is invalid if it changes declared cutoff, precision, access control, or failure semantics.

### 10.1 Result application notes
**Result note R-01 — Order invariance:** Positive rescaling preserves ordering, but null encoding, category treatment, precision, and illegal timestamps remain separate problems.
**Question R-01:** What would have to be true for this result not to apply to the proposed transform? Result record: R-01.
**Use R-01:** Write that answer into the feature validity condition before comparing model metrics.
**Result note R-02 — Smoothing:** Choose alpha through leakage-safe validation and support inspection; alpha is a fitted modelling choice, not a category property.
**Question R-02:** What would have to be true for this result not to apply to the proposed transform? Result record: R-02.
**Use R-02:** Write that answer into the feature validity condition before comparing model metrics.
**Result note R-03 — Dimensionality:** Count statistical and systems dimensions: each coordinate consumes support, artifact space, model capacity, and monitoring attention.
**Question R-03:** What would have to be true for this result not to apply to the proposed transform? Result record: R-03.
**Use R-03:** Write that answer into the feature validity condition before comparing model metrics.
**Result note R-04 — Determinism:** A deterministic transform can be unreproducible if lookup map, timezone library, seed, or input ordering is not fixed.
**Question R-04:** What would have to be true for this result not to apply to the proposed transform? Result record: R-04.
**Use R-04:** Write that answer into the feature validity condition before comparing model metrics.
**Result note R-05 — Validation:** Temporal validation exposes freshness, taxonomy drift, and future-information assumptions that random splits can hide.
**Question R-05:** What would have to be true for this result not to apply to the proposed transform? Result record: R-05.
**Use R-05:** Write that answer into the feature validity condition before comparing model metrics.
**Result note R-06 — Causality:** A predictive feature need not be causal, but causal plausibility can help identify unstable proxies and post-outcome variables.
**Question R-06:** What would have to be true for this result not to apply to the proposed transform? Result record: R-06.
**Use R-06:** Write that answer into the feature validity condition before comparing model metrics.
**Result note R-07 — Compression:** A low-dimensional encoding is lossy; decide whether discarded distinctions are irrelevant to the permitted decision, not merely training loss.
**Question R-07:** What would have to be true for this result not to apply to the proposed transform? Result record: R-07.
**Use R-07:** Write that answer into the feature validity condition before comparing model metrics.
**Result note R-08 — Monitoring:** Drift statistics require a reference with the same transform version; raw and transformed drift answer different questions.
**Question R-08:** What would have to be true for this result not to apply to the proposed transform? Result record: R-08.
**Use R-08:** Write that answer into the feature validity condition before comparing model metrics.
**Result note R-09 — Fairness:** A representation can create disparate performance through missingness, granularity, or proxy structure without an explicit protected field.
**Question R-09:** What would have to be true for this result not to apply to the proposed transform? Result record: R-09.
**Use R-09:** Write that answer into the feature validity condition before comparing model metrics.
**Result note R-10 — Reproducibility:** A reproduced number is insufficient unless it came from the same legal historical inputs rather than corrected future backfills.
**Question R-10:** What would have to be true for this result not to apply to the proposed transform? Result record: R-10.
**Use R-10:** Write that answer into the feature validity condition before comparing model metrics.

## 12. Quick Reference Card
| Field | Decision-ready reference |
|---|---|
| Objective | Convert admissible raw observations into reproducible model inputs for a stated decision cutoff. |
| Availability invariant | Every contributing fact and fitted artifact must be available no later than c_i. |
| Fit invariant | Learn scalers, bins, vocabularies, counts, and target encodings on permitted training data only. |
| Time rule | Use half-open trailing windows and distinguish event time from availability time. |
| Categorical choice | One-hot for moderate auditable vocabularies; hash for bounded width with collision acceptance; target encode only with fold isolation. |
| Numeric choice | Preserve units, scale only when learner geometry needs it, and guard log, ratio, and variance boundaries. |
| Missingness | Retain meaningful absence with an indicator and document null, empty, invalid, and fallback semantics. |
| Production proof | Replay historical cutoffs, compare training and serving fixtures, and monitor freshness, schema, support, and unknown rate. |
| Capacity check | Estimate rows × encoded width × bytes, plus state, latency headroom, artifact, and recovery costs. |
| Stop condition | Reject a feature when legality, semantic stability, support, governance, or capacity cannot be demonstrated. |
