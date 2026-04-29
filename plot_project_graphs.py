# import matplotlib.pyplot as plt
# import numpy as np

# # -----------------------------
# # Style
# # -----------------------------
# plt.style.use("seaborn-v0_8-whitegrid")
# TITLE_SIZE = 14
# LABEL_SIZE = 11
# TICK_SIZE = 10

# # -----------------------------
# # Data
# # -----------------------------
# methods_creation = ["pageIndex", "Vector Embedding"]
# creation_latency_sec = [32, 118]

# methods_retrieval = ["Vector Embedding", "pageIndex"]
# retrieval_latency_ms = [145, 320]

# metrics = ["Correctness", "Groundedness", "Completeness"]
# pageindex_scores = [3.1, 2.9, 3.0]
# vector_scores = [4.0, 3.8, 3.9]

# ft_metrics = ["Correctness", "Groundedness", "Completeness", "Avg User Rating"]
# before_ft = [3.8, 3.6, 3.7, 3.5]
# after_ft = [4.3, 4.1, 4.2, 4.0]

# # -----------------------------
# # Plot all in one figure (2x2)
# # -----------------------------
# fig, axs = plt.subplots(2, 2, figsize=(14, 10))
# fig.suptitle("LexQuery: Performance & Quality Comparison", fontsize=16, fontweight="bold")

# # Add outer whitespace around the full canvas
# fig.subplots_adjust(
#     left=0.08,   # empty space on left
#     right=0.95,  # empty space on right
#     top=0.90,    # empty space on top
#     bottom=0.10, # empty space on bottom
#     wspace=0.28, # horizontal gap between plots
#     hspace=0.35  # vertical gap between plots
# )

# # Graph 1: Creation Latency
# ax = axs[0, 0]
# bars = ax.bar(methods_creation, creation_latency_sec, color=["#4CAF50", "#1E88E5"])
# ax.set_title("1) Index Creation Latency", fontsize=TITLE_SIZE)
# ax.set_ylabel("Time (seconds)", fontsize=LABEL_SIZE)
# ax.tick_params(axis="both", labelsize=TICK_SIZE)
# ax.margins(x=0.18, y=0.15)  # inner whitespace around bars
# for b in bars:
#     ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{b.get_height():.0f}s",
#             ha="center", va="bottom", fontsize=10)

# # Graph 2: Retrieval Latency
# ax = axs[0, 1]
# bars = ax.bar(methods_retrieval, retrieval_latency_ms, color=["#1E88E5", "#4CAF50"])
# ax.set_title("2) Retrieval Latency", fontsize=TITLE_SIZE)
# ax.set_ylabel("Time (ms)", fontsize=LABEL_SIZE)
# ax.tick_params(axis="both", labelsize=TICK_SIZE)
# ax.margins(x=0.18, y=0.15)
# for b in bars:
#     ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 5, f"{b.get_height():.0f}ms",
#             ha="center", va="bottom", fontsize=10)

# # Graph 3: Review Quality (grouped bars)
# ax = axs[1, 0]
# x = np.arange(len(metrics))
# width = 0.35
# bars1 = ax.bar(x - width / 2, pageindex_scores, width, label="pageIndex", color="#4CAF50")
# bars2 = ax.bar(x + width / 2, vector_scores, width, label="Vector Embedding", color="#1E88E5")
# ax.set_title("3) Review Scores (Feedback Loop)", fontsize=TITLE_SIZE)
# ax.set_ylabel("Score (out of 5)", fontsize=LABEL_SIZE)
# ax.set_xticks(x)
# ax.set_xticklabels(metrics)
# ax.set_ylim(0, 5)
# ax.legend()
# ax.tick_params(axis="both", labelsize=TICK_SIZE)
# ax.margins(x=0.20, y=0.12)
# for bars in [bars1, bars2]:
#     for b in bars:
#         ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05, f"{b.get_height():.1f}",
#                 ha="center", va="bottom", fontsize=9)

# # Graph 4: Fine-tuning Improvement
# ax = axs[1, 1]
# x2 = np.arange(len(ft_metrics))
# ax.plot(x2, before_ft, marker="o", linewidth=2.2, label="Before Fine-Tuning", color="#FB8C00")
# ax.plot(x2, after_ft, marker="o", linewidth=2.2, label="After Fine-Tuning", color="#8E24AA")
# ax.set_title("4) Fine-Tuning Impact (Illustrative)", fontsize=TITLE_SIZE)
# ax.set_ylabel("Score (out of 5)", fontsize=LABEL_SIZE)
# ax.set_xticks(x2)
# ax.set_xticklabels(ft_metrics, rotation=15)
# ax.set_ylim(0, 5)
# ax.legend()
# ax.tick_params(axis="both", labelsize=TICK_SIZE)
# ax.margins(x=0.12, y=0.12)

# # Save with extra border padding for screenshot-friendly output
# plt.savefig("lexquery_comparison_graphs.png", dpi=300, bbox_inches="tight", pad_inches=0.4)
# plt.show()

# print("Saved: lexquery_comparison_graphs.png")

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-whitegrid")

# -----------------------------
# Data
# -----------------------------
methods_creation = ["pageIndex", "Vector Embedding"]
creation_latency_sec = [32, 118]

methods_retrieval = ["Vector Embedding", "pageIndex"]
retrieval_latency_ms = [145, 320]

metrics = ["Correctness", "Groundedness", "Completeness"]
pageindex_scores = [3.1, 2.9, 3.0]
vector_scores = [4.0, 3.8, 3.9]

ft_metrics = ["Correctness", "Groundedness", "Completeness", "Avg User Rating"]
before_ft = [3.8, 3.6, 3.7, 3.5]
after_ft = [4.3, 4.1, 4.2, 4.0]


def save_fig(filename: str):
    plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0.5)
    plt.close()
    print(f"Saved: {filename}")


# 1) Index Creation Latency
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(methods_creation, creation_latency_sec, color=["#4CAF50", "#1E88E5"])
ax.set_title("Index Creation Latency")
ax.set_ylabel("Time (seconds)")
ax.margins(x=0.20, y=0.15)
for b in bars:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{b.get_height():.0f}s",
            ha="center", va="bottom")
save_fig("graph1_index_creation_latency.png")

# 2) Retrieval Latency
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(methods_retrieval, retrieval_latency_ms, color=["#1E88E5", "#4CAF50"])
ax.set_title("Retrieval Latency")
ax.set_ylabel("Time (ms)")
ax.margins(x=0.20, y=0.15)
for b in bars:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 5, f"{b.get_height():.0f}ms",
            ha="center", va="bottom")
save_fig("graph2_retrieval_latency.png")

# 3) Review Scores (Feedback Loop)
fig, ax = plt.subplots(figsize=(8, 6))
x = np.arange(len(metrics))
width = 0.35
bars1 = ax.bar(x - width / 2, pageindex_scores, width, label="pageIndex", color="#4CAF50")
bars2 = ax.bar(x + width / 2, vector_scores, width, label="Vector Embedding", color="#1E88E5")
ax.set_title("Review Scores (Feedback Loop)")
ax.set_ylabel("Score (out of 5)")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 5)
ax.legend()
ax.margins(x=0.20, y=0.12)
for bars in [bars1, bars2]:
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05, f"{b.get_height():.1f}",
                ha="center", va="bottom")
save_fig("graph3_review_scores.png")

# 4) Fine-Tuning Impact (Illustrative)
fig, ax = plt.subplots(figsize=(8, 6))
x2 = np.arange(len(ft_metrics))
ax.plot(x2, before_ft, marker="o", linewidth=2.2, label="Before Fine-Tuning", color="#FB8C00")
ax.plot(x2, after_ft, marker="o", linewidth=2.2, label="After Fine-Tuning", color="#8E24AA")
ax.set_title("Fine-Tuning Impact (Illustrative)")
ax.set_ylabel("Score (out of 5)")
ax.set_xticks(x2)
ax.set_xticklabels(ft_metrics, rotation=15)
ax.set_ylim(0, 5)
ax.legend()
ax.margins(x=0.12, y=0.12)
save_fig("graph4_finetuning_impact.png")