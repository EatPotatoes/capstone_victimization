# %%
import pandas as pd

dataset = pd.read_csv("/Users/nicoletimko/Desktop/capstone_victimization/full_dataset/full_data.csv")


# %%
victim_cols = ["violent_victim_2002", "violent_victim_2007", "violent_victim_2008", "violent_victim_2009", "violent_victim_2013"]

victim_data = dataset[victim_cols].apply(pd.to_numeric, errors="coerce")
reported_yes = (victim_data == 1).any(axis=1)
has_response = ((victim_data == 0) | (victim_data == 1)).any(axis=1)

dataset["victim_ever"] = pd.Series(pd.NA, index=dataset.index, dtype="Int64") # fills everything NA first
# so if not a valid response or 1 then will stay NA
dataset.loc[has_response, "victim_ever"] = 0
dataset.loc[reported_yes, "victim_ever"] = 1

# %%
victim_age_cols = [
    "age_at_violent_victimization_2002",
    "age_at_violent_victimization_2007",
    "age_at_violent_victimization_2008",
    "age_at_violent_victimization_2009",
    "age_at_violent_victimization_2013"
]

def collect_victimization_ages(row):
    ages = []

    for col in victim_age_cols:
        age = row[col]

        # Keep valid reported ages only
        if pd.notna(age) and age > 0:
            ages.append(int(age))

    return ages


dataset["victimization_ages"] = dataset.apply(
    collect_victimization_ages,
    axis=1
)

# %%
dataset["ever_enrolled"] = ((dataset["college_history"] == "Enrolled") | (dataset["college_history"] == "Grad program only")).astype(int)

# %% [markdown]
# Did we want grad program?

# %%
dataset["college_history"].value_counts()



# %%
def create_outcome(row):
    ages = row["victimization_ages"]
    enrolled = row["ever_enrolled"]
    enrollment_age = row["age_at_first_college"]
    victim_ever = row["victim_ever"]

    # Missing victimization information
    if pd.isna(victim_ever):
        return pd.Series({
            "Y": pd.NA,
            "prior_victimization": pd.NA,
            "same_age_as_enrollment": pd.NA,

        })

    # Never reported victimization
    if victim_ever == 0:
        return pd.Series({
            "Y": 0,
            "prior_victimization": 0,
            "same_age_as_enrollment": 0,

        })

    # Make sure victimization ages are stored as a list
    if not isinstance(ages, list):
        valid_ages = []

    else:
        valid_ages = [
            age for age in ages
            if pd.notna(age) and age > 0
        ]

    # Reported victimization, but no valid age is available
    if len(valid_ages) == 0:
        return pd.Series({
            "Y": pd.NA,
            "prior_victimization": pd.NA,
            "same_age_as_enrollment": pd.NA,
        })

    # Missing college-enrollment status
    if pd.isna(enrolled):
        return pd.Series({
            "Y": pd.NA,
            "prior_victimization": pd.NA,
            "same_age_as_enrollment": pd.NA,
        })

    # --------------------------------------------------
    # Never enrolled
    # --------------------------------------------------

    if enrolled == 0:

        # Victimized during the outcome period
        during_18_24 = any(
            18 <= age <= 24
            for age in valid_ages
        )

        # For never-enrolled people, prior means before age 17
        before_18 = any(
            age < 18
            for age in valid_ages
        )

        return pd.Series({
            "Y": int(during_18_24),
            "prior_victimization": int(before_18),
            "same_age_as_enrollment": 0,
        })

    # --------------------------------------------------
    # Enrolled
    # --------------------------------------------------

    # Enrolled, but first enrollment age is unavailable
    if pd.isna(enrollment_age):
        return pd.Series({
            "Y": pd.NA,
            "prior_victimization": pd.NA,
            "same_age_as_enrollment": pd.NA,
        })

    # Victimization before first enrollment
    prior = any(
        age < enrollment_age
        for age in valid_ages
    )

    # Victimization at the same age as first enrollment
    same_age = any(
        age == enrollment_age
        for age in valid_ages
    )

    # Victimization after enrollment during ages 17–24
    after_enrollment_18_24 = any(
        age > enrollment_age
        and 18 <= age <= 24
        for age in valid_ages
    )

    # Definite qualifying outcome
    if after_enrollment_18_24:
        outcome = 1
    # Victimization only known to be at the same age
    elif same_age:
        outcome = pd.NA

    else:
        outcome = 0

    return pd.Series({
        "Y": outcome,
        "prior_victimization": int(prior),
        "same_age_as_enrollment": int(same_age),
    })

# %%
outcome_columns = [
    "Y",
    "prior_victimization",
    "same_age_as_enrollment",
]

dataset[outcome_columns] = dataset.apply(
    create_outcome,
    axis=1
)

for col in outcome_columns:
    dataset[col] = dataset[col].astype("Int64")


# %%
dataset.to_csv("/Users/nicoletimko/Desktop/capstone_victimization/full_dataset/data_with_Y_18.csv")


