import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda_analysis(clean_csv_path="data/processed/clean_water_quality.csv", figures_dir="reports/figures", summary_dir="reports"):
    """
    Performs Exploratory Data Analysis (EDA) on environmental parameters and object detection classes:
    - Calculates summary statistics and correlations
    - Saves high-resolution visualization charts
    - Exports EDA JSON metadata summary
    """
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)
    
    df = pd.read_csv(clean_csv_path)
    
    # 1. Statistical Summary
    num_cols = [
        "water_temp_c", "ph_level", "turbidity_ntu", "dissolved_oxygen_mg_l",
        "salinity_psu", "microplastic_particles_m3", "macroplastic_count_km2",
        "coastal_proximity_km", "wave_height_m", "ocean_current_speed_m_s",
        "industrial_discharge_score", "pollution_index"
    ]
    stats_summary = df[num_cols].describe().to_dict()
    
    # 2. Correlation Matrix Plot
    plt.figure(figsize=(12, 9))
    sns.set_theme(style="white")
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8}, annot=True, fmt=".2f", annot_kws={"size": 8})
    plt.title("Environmental Water Quality & Pollution Correlation Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    corr_path = os.path.join(figures_dir, "correlation_matrix.png")
    plt.savefig(corr_path, dpi=300)
    plt.close()
    
    # 3. Class Distribution Bar Chart
    plt.figure(figsize=(8, 5))
    order = ["Low", "Medium", "High", "Critical"]
    palette = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
    ax = sns.countplot(data=df, x="pollution_level", order=order, palette=palette)
    plt.title("Pollution Risk Level Class Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Pollution Risk Level", fontsize=11)
    plt.ylabel("Sample Count", fontsize=11)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight="bold", xytext=(0, 5), textcoords='offset points')
    plt.tight_layout()
    class_path = os.path.join(figures_dir, "class_distribution.png")
    plt.savefig(class_path, dpi=300)
    plt.close()

    # 4. Feature Distributions Grid
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    features_to_plot = [
        "turbidity_ntu", "microplastic_particles_m3", "macroplastic_count_km2",
        "dissolved_oxygen_mg_l", "industrial_discharge_score", "coastal_proximity_km",
        "ph_level", "water_temp_c", "pollution_index"
    ]
    for idx, feature in enumerate(features_to_plot):
        row, col = idx // 3, idx % 3
        sns.histplot(df[feature], kde=True, ax=axes[row, col], color="#3498db")
        axes[row, col].set_title(f"Distribution of {feature}", fontsize=11, fontweight="bold")
        axes[row, col].set_xlabel("")
    plt.tight_layout()
    dist_path = os.path.join(figures_dir, "environmental_distributions.png")
    plt.savefig(dist_path, dpi=300)
    plt.close()

    # 5. Waste / Pollution Index vs Coastal Proximity Scatter Plot
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=df, x="coastal_proximity_km", y="pollution_index", hue="pollution_level",
                    hue_order=order, palette=palette, alpha=0.7, s=60)
    plt.title("Pollution Index vs Coastal Proximity (km)", fontsize=14, fontweight="bold")
    plt.xlabel("Distance to Shoreline (km)", fontsize=11)
    plt.ylabel("Pollution Risk Index (0-100)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    scatter_path = os.path.join(figures_dir, "waste_by_location.png")
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    
    # 6. Analyze Object Detection Annotation Frequencies
    lbl_files = glob.glob("data/raw/labels/*.txt")
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    class_names = {0: "plastic_bottle", 1: "plastic_bag", 2: "fishing_net", 3: "other_waste"}
    
    for f in lbl_files:
        with open(f, "r") as file_in:
            for line in file_in:
                parts = line.strip().split()
                if len(parts) >= 1:
                    cls_id = int(parts[0])
                    if cls_id in class_counts:
                        class_counts[cls_id] += 1

    plt.figure(figsize=(8, 5))
    names = [class_names[k] for k in sorted(class_counts.keys())]
    counts = [class_counts[k] for k in sorted(class_counts.keys())]
    plt.bar(names, counts, color="#9b59b6")
    plt.title("Deep Learning Object Detection Class Bounding Box Counts", fontsize=14, fontweight="bold")
    plt.xlabel("Waste Object Class", fontsize=11)
    plt.ylabel("Total Bounding Box Instances", fontsize=11)
    plt.tight_layout()
    dl_dist_path = os.path.join(figures_dir, "dl_class_distribution.png")
    plt.savefig(dl_dist_path, dpi=300)
    plt.close()

    # Save summary metadata
    summary_data = {
        "dataset_sample_size": len(df),
        "pollution_level_counts": df["pollution_level"].value_counts().to_dict(),
        "mean_pollution_index": float(df["pollution_index"].mean()),
        "max_microplastic_particles_m3": float(df["microplastic_particles_m3"].max()),
        "top_correlated_with_pollution_index": corr["pollution_index"].abs().sort_values(ascending=False).head(5).to_dict(),
        "dl_bounding_box_counts": {class_names[k]: class_counts[k] for k in class_counts}
    }
    
    with open(os.path.join(summary_dir, "eda_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=2)

    print("EDA & Data Analysis Completed Successfully!")

if __name__ == "__main__":
    run_eda_analysis()
