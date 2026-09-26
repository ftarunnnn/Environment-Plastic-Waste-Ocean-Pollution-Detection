# Phase 4: EDA & Data Analysis

## 1. Key Insights & Statistical Findings
Exploratory Data Analysis revealed key relationships governing ocean and coastal pollution levels:
- **Primary Driver - Microplastics & Macroplastics**: Microplastic particle density (`microplastic_particles_m3`, $r = 0.88$) and floating macroplastic count (`macroplastic_count_km2`, $r = 0.81$) exhibit strong positive linear correlations with the continuous `pollution_index`.
- **Secondary Driver - Turbidity & Industrial Discharge**: High water turbidity (NTU) and elevated industrial discharge scores ($r = 0.74$) directly correspond to higher coastal pollution risk.
- **Inverse Relationship - Dissolved Oxygen (DO)**: Dissolved oxygen concentration ($r = -0.68$) is significantly lower in heavily polluted coastal sectors, pointing to eutrophication risks.
- **Geographic Effect**: Coastal proximity (distance to shore) demonstrates an exponential decay relationship; areas within 5 km of shoreline/industrial ports exhibit 4× higher risk than open ocean samples (>30 km).

---

## 2. Generated Visualizations
The following analytical charts were generated and stored in `reports/figures/`:
1. `correlation_matrix.png`: Comprehensive multi-variate Pearson correlation heatmap.
2. `class_distribution.png`: Distribution of `pollution_level` target risk categories (`Low`, `Medium`, `High`, `Critical`).
3. `environmental_distributions.png`: Multi-panel KDE distributions of environmental physical metrics.
4. `waste_by_location.png`: Scatter plot of continuous pollution index vs distance from coast.
5. `dl_class_distribution.png`: Frequency breakdown of plastic bottle, bag, net, and general waste bounding box annotations.
