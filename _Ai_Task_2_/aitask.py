import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Setup: 5x5 matrix & 16 students
M = np.random.randint(10, 100, (5, 5))
df = pd.DataFrame(
{
"name": [f"S{i}" for i in range(16)],
"subject": np.random.choice(["Math", "Physics", "Chemistry"], 16),
"marks": np.random.choice([np.nan, 45, 60, 75, 88, 92], 16),
"attendance": np.random.randint(55, 100, 16),
}
)

# (i) Matrix: transpose, row/col sums, max position
M_T, r_sum, c_sum = M.T, M.sum(axis=1), M.sum(axis=0)
max_pos = np.unravel_index(np.argmax(M), M.shape)

# (ii) DataFrame analysis
df["marks"] = df["marks"].fillna(df["marks"].mean())
avg_sub = df.groupby("subject")["marks"].mean()
low_att = df[df["attendance"] < 75]
df["result"] = np.where(df["marks"] >= 50, "Pass", "Fail")
top5 = df.nlargest(5, "marks")

# (iii) Visualizations
df["marks"].plot(
kind="line",
color="crimson",
marker="o",
title="Marks Trend",
grid=True,
legend=True,
)
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
avg_sub.plot(kind="bar", ax=ax1, color="teal", title="Avg Marks")
df["attendance"].plot(
kind="hist", ax=ax2, bins=5, color="coral", title="Attendance"
)
plt.tight_layout()
plt.show()