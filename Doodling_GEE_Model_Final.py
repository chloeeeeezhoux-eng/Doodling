import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ============================================================
# GEE RECALL ANALYSIS: SIX UNIQUE PASSAGES
# ============================================================
# Design note:
#   Visual passages 1-3 and auditory passages 1-3 are different texts.
#   Therefore, task modality is confounded with passage identity. This script
#   controls for each of the six passages and tests doodling-condition effects,
#   but DOES NOT estimate an overall visual-versus-auditory main effect.
# ============================================================

# ------------------------------------------------------------
# 1) Read data
# ------------------------------------------------------------
df = pd.read_csv("/Users/xiaoyuzhou/Desktop/Doodling/Doodling Result.csv")
key = pd.read_excel("/Users/xiaoyuzhou/Desktop/Doodling/recall_answer_key.xlsx")

# ------------------------------------------------------------
# 2) Define and score recall items
# ------------------------------------------------------------
visual_cols = [f"visual_{i}" for i in range(1, 13)]
audio_cols = [f"audio_{i}" for i in range(1, 13)]
all_recall_cols = visual_cols + audio_cols

df = df.dropna(subset=all_recall_cols).copy()

answer_key = key.set_index("item")["correct_answer"]

df.loc[:, visual_cols] = df[visual_cols].eq(
    answer_key.reindex(visual_cols), axis=1
).astype(int)
df.loc[:, audio_cols] = df[audio_cols].eq(
    answer_key.reindex(audio_cols), axis=1
).astype(int)


# Group number to account for fatigue

df["group_number"] = df["group_number"].astype(str).str.strip()

task_order_map = {
    "1":"visual_first",
    "2":"auditory_first",
}

df["task_order"] = df["group_number"].map(task_order_map)

# validation check
if df["task_order"].isna().any():
    raise ValueError(
        "Some group_number values were not mapped. "
        "Check group_number and task_order_map."
    )

# ------------------------------------------------------------
# 3) Map counterbalanced condition versions to passages
# ------------------------------------------------------------
visual_condition_map = {
    "A": {1: "free", 2: "structured", 3: "control"},
    "B": {1: "control", 2: "free", 3: "structured"},
    "C": {1: "structured", 2: "control", 3: "free"},
    "D": {1: "control", 2: "structured", 3: "free"},
}

auditory_condition_map = {
    "A": {1: "free", 2: "structured", 3: "control"},
    "B": {1: "control", 2: "free", 3: "structured"},
    "C": {1: "structured", 2: "control", 3: "free"},
    "D": {1: "free", 2: "control", 3: "structured"},
}

def make_task_long(data, task_name, task_type_value, score_cols,
                   condition_col, condition_map):
    """Convert one task's 12 recall items to item-level long format."""
    long_df = data.melt(
        id_vars=["p_id", condition_col, "task_order"],
        value_vars=score_cols,
        var_name="item",
        value_name="score",
    ).copy()

    # Questions 1-4 belong to passage 1, 5-8 to passage 2, and 9-12 to passage 3.
    long_df["QuestionNum"] = long_df["item"].str.extract(r"(\d+)").astype(int)
    long_df["PassageIndex"] = ((long_df["QuestionNum"] - 1) // 4) + 1
    long_df["TaskName"] = task_name
    long_df["TaskType"] = task_type_value  # Auditory = 0; Visual = 1

    long_df["DoodleType"] = [
        condition_map[condition][passage]
        for condition, passage in zip(
            long_df[condition_col], long_df["PassageIndex"]
        )
    ]

    return long_df[
        [
            "p_id", "TaskName", "TaskType", "QuestionNum", "PassageIndex",
            "DoodleType", "score","task_order"
        ]
    ]


# ------------------------------------------------------------
# 4) Create the analysis data set
# ------------------------------------------------------------
visual_long = make_task_long(
    df, "Visual", 1, visual_cols, "visual_condition", visual_condition_map
)
auditory_long = make_task_long(
    df, "Auditory", 0, audio_cols, "auditory_condition", auditory_condition_map
)

gee_df = pd.concat([visual_long, auditory_long], ignore_index=True)
gee_df["p_id"] = gee_df["p_id"].astype(str)
gee_df["score"] = gee_df["score"].astype(int)


gee_df["PassageID"] = (
    gee_df["TaskName"] + "_Passage_" + gee_df["PassageIndex"].astype(str)
)
gee_df["PassageID"] = pd.Categorical(
    gee_df["PassageID"],
    categories=[
        "Auditory_Passage_1", "Auditory_Passage_2", "Auditory_Passage_3",
        "Visual_Passage_1", "Visual_Passage_2", "Visual_Passage_3",
    ],
)
gee_df["DoodleType"] = pd.Categorical(
    gee_df["DoodleType"], categories=["control", "free", "structured"]
)

# Block effect
gee_df["BlockPosition"] = np.where(
    (
        ((gee_df["task_order"] == "visual_first") &
         (gee_df["TaskName"] == "Visual"))
        |
        ((gee_df["task_order"] == "auditory_first") &
         (gee_df["TaskName"] == "Auditory"))
    ),
    "first",
    "second",
)

# These interaction indicators compare visual and auditory doodling effects.
gee_df["VisualFree"] = (
    (gee_df["TaskType"] == 1) & (gee_df["DoodleType"] == "free")
).astype(int)
gee_df["VisualStructured"] = (
    (gee_df["TaskType"] == 1) & (gee_df["DoodleType"] == "structured")
).astype(int)

print("\nNumber of participants:", gee_df["p_id"].nunique())
print("Number of item-level observations:", len(gee_df))
print("\nCondition counts by unique passage:\n")
print(pd.crosstab(gee_df["PassageID"], gee_df["DoodleType"]))

# ------------------------------------------------------------
# 5) Primary GEE model
# ------------------------------------------------------------

# Interpretation of doodling terms:
#   free / structured: effect versus control in AUDITORY trials.
#   VisualFree: whether the free-doodling effect differs in VISUAL trials.
#   VisualStructured: whether the structured-doodling effect differs in VISUAL
#                     trials.
formula = (
    "score ~ C(PassageID) "
    "+ C(DoodleType, Treatment(reference='control')) "
    "+ VisualFree + VisualStructured"
    "+ C(BlockPosition)"
)

gee_final = smf.gee(
    formula,
    groups="p_id",
    data=gee_df,
    family=sm.families.Binomial(),
    cov_struct=sm.cov_struct.Exchangeable(),
).fit()

print("\n================ GEE: SIX-PASSAGE FIXED-EFFECT MODEL ================\n")
print(gee_final.summary())

# ------------------------------------------------------------
# 6) Planned contrasts for condition effects within each modality
# ------------------------------------------------------------
free_term = "C(DoodleType, Treatment(reference='control'))[T.free]"
structured_term = "C(DoodleType, Treatment(reference='control'))[T.structured]"


def print_contrast(result, label, weights):
    """Print a Wald test for a named linear combination of coefficients."""
    contrast = np.zeros(len(result.params))
    for term, weight in weights.items():
        contrast[result.params.index.get_loc(term)] = weight
    print(f"\n{label}")
    print(result.t_test(contrast))


print("\n================ PLANNED DOODLING CONTRASTS ================")
print_contrast(
    gee_final,
    "Auditory: free doodling versus control",
    {free_term: 1},
)
print_contrast(
    gee_final,
    "Auditory: structured doodling versus control",
    {structured_term: 1},
)
print_contrast(
    gee_final,
    "Visual: free doodling versus control",
    {free_term: 1, "VisualFree": 1},
)
print_contrast(
    gee_final,
    "Visual: structured doodling versus control",
    {structured_term: 1, "VisualStructured": 1},
)

# ------------------------------------------------------------
# 7) Descriptive means (unadjusted)
# ------------------------------------------------------------
print("\n================ DESCRIPTIVE MEANS BY TASK AND CONDITION ================\n")
print(
    gee_df.groupby(["TaskName", "DoodleType"], observed=False)["score"]
    .agg(["mean", "count"])
    .reset_index()
)

print("\n================ DESCRIPTIVE MEANS BY UNIQUE PASSAGE ================\n")
print(
    gee_df.groupby(["PassageID", "DoodleType"], observed=False)["score"]
    .agg(["mean", "count"])
    .reset_index()
)