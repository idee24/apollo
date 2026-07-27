# Feature Glossary - Apollo Model A

Plain-language meaning of the 15 features Model A uses to estimate the probability
of ≥1 fatality. These are **schema definitions** (what an input means), not GTD
incident data. Machine-readable labels for the low-cardinality codes live in
[`engine/codes.py`](../engine/codes.py). Feature classes (PRE_EVENT / SCENARIO)
are defined in [`prediction_time_contract.md`](prediction_time_contract.md); no
outcome field is ever a feature.

## What each feature means

- **iyear, imonth** - incident year and month. Year carries slow trend and era
  effects; month captures seasonality. (`imonth` 0 = unknown → treated as missing.)
- **latitude, longitude, specificity, vicinity** - where the incident occurs and how
  precisely it is geocoded. `specificity` runs 1 (exact location) to 5 (only a region
  known); `vicinity` flags whether it happened in a city or its immediate surroundings.
- **country, region** - location codes. `region` is one of 12 world regions; `country`
  is a finer code. Region is the more stable, higher-level signal.
- **attacktype1** - the primary tactic (e.g. bombing, armed assault, kidnapping).
- **targtype1, targsubtype1** - what was targeted (e.g. private citizens, military,
  government) and a finer subtype.
- **natlty1** - nationality of the target.
- **weaptype1, weapsubtype1** - the weapon category (e.g. explosives, firearms,
  incendiary) and a finer subtype.
- **suicide** - whether the attack was a suicide attack (0/1).

## How to read a prediction

Model A returns a **calibrated probability that the described incident results in one
or more fatalities**, with an uncertainty band reflecting disagreement across
calibration folds. It is retrospective and probabilistic: it describes how lethal
*similar historical incidents* were, not whether an attack will occur. Weapon and
attack type are among the strongest signals of lethality; precise location and
timing matter less. See [`intended_use.md`](intended_use.md) for scope limits and
[`model_card.md`](model_card.md) for measured accuracy, calibration and subgroup
behaviour - including that mid-range probabilities (0.4–0.7) are over-confident on
recent data, and that the model over-predicts fatality in some regions.

## Code tables (low-cardinality features)

**region (1-12):** 1 North America, 2 Central America & Caribbean, 3 South America,
4 East Asia, 5 Southeast Asia, 6 South Asia, 7 Central Asia, 8 Western Europe,
9 Eastern Europe, 10 Middle East & North Africa, 11 Sub-Saharan Africa,
12 Australasia & Oceania.

**attacktype1 (1-9):** 1 Assassination, 2 Armed Assault, 3 Bombing/Explosion,
4 Hijacking, 5 Hostage Taking (Barricade), 6 Hostage Taking (Kidnapping),
7 Facility/Infrastructure Attack, 8 Unarmed Assault, 9 Unknown.

**weaptype1 (1-13):** 1 Biological, 2 Chemical, 3 Radiological, 4 Nuclear,
5 Firearms, 6 Explosives, 7 Fake Weapons, 8 Incendiary, 9 Melee,
10 Vehicle (non-explosive), 11 Sabotage Equipment, 12 Other, 13 Unknown.

**targtype1 (1-22):** 1 Business, 2 Government (General), 3 Police, 4 Military,
5 Abortion Related, 6 Airports & Aircraft, 7 Government (Diplomatic),
8 Educational Institution, 9 Food or Water Supply, 10 Journalists & Media,
11 Maritime, 12 NGO, 13 Other, 14 Private Citizens & Property,
15 Religious Figures/Institutions, 16 Telecommunication, 17 Terrorists/Non-State
Militia, 18 Tourists, 19 Transportation, 20 Unknown, 21 Utilities,
22 Violent Political Party.

**suicide:** 0 no, 1 yes.
