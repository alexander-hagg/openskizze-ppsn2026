# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import os
import glob
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import yaml
from scipy.stats import mannwhitneyu
import logging

# PyLaTeX for report generation
from pylatex import Document, Section, Figure, Table, Subsection, NoEscape

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

KLAM_DIR = "results/sail_klam"
FF_DIR = "results/sail_flood_fill"
REPORT_DIR = "analysis_report"
os.makedirs(REPORT_DIR, exist_ok=True)


def find_archive_paths(directory: str) -> list:
    """Finds all final QD archive paths from a results directory."""
    logging.info(f"Searching for archives in: {directory}")
    archive_paths = glob.glob(f"{directory}/rep_*/**/*FinalQD_archive.pkl", recursive=True)
    logging.info(f"Found {len(archive_paths)} archive paths.")
    return sorted(archive_paths)

def extract_data_from_archive_paths(archive_paths: list) -> dict:
    """
    Extracts solutions (genomes), features (measures), phenotypes (heightmaps),
    and performance stats from a list of archive file paths.
    """
    all_solutions, all_phenotypes, all_features, all_objectives = [], [], [], []
    stats_list = []
    logging.info(f"Extracting data from {len(archive_paths)} archives...")

    for i, path in enumerate(archive_paths):
        try:
            with open(path, 'rb') as f:
                archive = pickle.load(f)
        except Exception as e:
            logging.warning(f"Could not load or process archive at {path}. Error: {e}")
            continue

        data = archive.data(fields=['solution', 'heightmaps', 'measures', 'objective'])
        
        num_elites = data['solution'].shape[0]
        if num_elites > 0:
            all_solutions.append(data['solution'])
            all_phenotypes.append(data['heightmaps'])
            all_features.append(data['measures'])
            all_objectives.append(data['objective'])
        logging.info(f"  - Archive {i+1} ({os.path.basename(path)}): Found {num_elites} elites.")
        
        stats_list.append({
            "QD Score": archive.stats.qd_score,
            "Coverage (%)": archive.stats.coverage * 100
        })
            
    return {
        'solutions': np.vstack(all_solutions) if all_solutions else np.array([]),
        'phenotypes': np.vstack(all_phenotypes) if all_phenotypes else np.array([]),
        'features': np.vstack(all_features) if all_features else np.array([]),
        'objectives': np.vstack(all_objectives) if all_objectives else np.array([]), # hstack?
        'stats_df': pd.DataFrame(stats_list)
    }

def perform_statistical_tests(klam_stats_df: pd.DataFrame, ff_stats_df: pd.DataFrame) -> str:
    """Performs Mann-Whitney U tests to compare performance metrics."""
    logging.info("Performing statistical analysis on performance metrics...")
    results = []
    
    qd_u, qd_p = mannwhitneyu(klam_stats_df['QD Score'], ff_stats_df['QD Score'], alternative='two-sided')
    results.append(f"QD Score: U-statistic={qd_u:.2f}, p-value={qd_p:.4f}")
    
    cov_u, cov_p = mannwhitneyu(klam_stats_df['Coverage (%)'], ff_stats_df['Coverage (%)'], alternative='two-sided')
    results.append(f"Coverage (%): U-statistic={cov_u:.2f}, p-value={cov_p:.4f}")
    
    conclusion = ("A p-value < 0.05 typically indicates a statistically significant difference "
                  "between the two methods for that metric.")
    
    full_report = "\n".join(results) + "\n\n" + conclusion
    logging.info("Statistical Analysis Results:\n" + full_report)
    return full_report

def generate_aggregate_heatmap(archive_paths: list, title: str, output_path: str):
    """Creates and saves a heatmap showing the average objective across all runs."""
    logging.info(f"Generating aggregate heatmap: {title}")
    if not archive_paths:
        logging.warning("No archive paths provided for heatmap. Skipping.")
        return
        
    all_elites_data = []
    archive_dims = None
    for path in archive_paths:
        try:
            with open(path, 'rb') as f:
                ar = pickle.load(f)
        except Exception as e:
            logging.warning(f"Could not load archive at {path} for heatmap. Error: {e}")
            continue
        
        if archive_dims is None:
            archive_dims = ar.dims

        data = ar.data(fields=['objective', 'index'])
        if data['objective'].shape[0] > 0:
            grid_indices = ar.int_to_grid_index(data['index'])
            df = pd.DataFrame({
                'objective': data['objective'],
                'index_0': grid_indices[:, 0],
                'index_1': grid_indices[:, 1]
            })
            all_elites_data.append(df)
    
    if not all_elites_data:
        logging.warning("No elite data found for heatmap. Skipping.")
        return

    all_elites = pd.concat(all_elites_data)
    heatmap_data = all_elites.groupby(['index_0', 'index_1'])['objective'].mean().unstack()
    
    aggregate_grid = np.full((archive_dims[0], archive_dims[1]), np.nan)
    
    for row_idx, row in heatmap_data.iterrows():
        for col_idx, value in row.items():
            aggregate_grid[int(row_idx), int(col_idx)] = value

    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap("viridis")
    cmap.set_bad(color='lightgrey')
    sns.heatmap(aggregate_grid, annot=False, cmap=cmap)

    with open("domain_description/cfg.yml") as f: config = yaml.safe_load(f)
    labels = [config['labels'][i] for i in config['features']]
    plt.xlabel(labels[1], fontsize=12); plt.ylabel(labels[0], fontsize=12)
    plt.title(title, fontsize=16)
    plt.savefig(output_path, dpi=150, bbox_inches='tight'); plt.close()
    logging.info(f"Saved aggregate heatmap to: {output_path}")

def generate_tsne_plot(klam_data: np.ndarray, ff_data: np.ndarray, data_type: str, output_path: str):
    """Generates and saves a t-SNE projection for genomes or phenotypes."""
    logging.info(f"Starting t-SNE projection for {data_type}s...")
    if klam_data.size == 0 or ff_data.size == 0:
        logging.warning(f"Not enough data for t-SNE plot of {data_type}s. Skipping.")
        return

    all_data = np.vstack([klam_data, ff_data])
    labels = np.array(['KLAM-21'] * len(klam_data) + ['Flood Fill'] * len(ff_data))
    flat_data = all_data.reshape(all_data.shape[0], -1)
    
    perplexity_val = min(30, flat_data.shape[0] - 1)
    logging.info(f"Running t-SNE with perplexity={perplexity_val} on {flat_data.shape[0]} samples...")
    
    tsne = TSNE(n_components=2, verbose=0, perplexity=perplexity_val, max_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(flat_data)
    
    df = pd.DataFrame({'t-SNE 1': tsne_results[:, 0], 't-SNE 2': tsne_results[:, 1], 'Method': labels})
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=df, x='t-SNE 1', y='t-SNE 2', hue='Method', palette='coolwarm', alpha=0.7, s=50)
    plt.title(f"t-SNE Projection of All Solution {data_type.capitalize()}s", fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(output_path, dpi=150, bbox_inches='tight'); plt.close()
    logging.info(f"Saved t-SNE plot to: {output_path}")

def generate_latex_report(klam_stats, ff_stats, stat_tests_results, figures):
    """Generates a PDF report summarizing the analysis."""
    logging.info("Generating LaTeX report...")
    doc = Document(geometry_options={"tmargin": "1in", "lmargin": "1in"})
    doc.preamble.append(NoEscape(r'\usepackage{graphicx}\usepackage{booktabs}\usepackage{verbatim}'))
    doc.append(NoEscape(r'\title{Analysis of SAIL Optimization Runs}\author{Automated Analysis Script}\date{\today}\maketitle'))
    
    with doc.create(Section('Performance Summary')):
        doc.append("This section compares performance metrics (QD Score, Coverage) across 10 replicate runs for each evaluation method.")
        with doc.create(Subsection('KLAM-21 Runs')):
            with doc.create(Table(position='h!')) as table:
                table.add_caption('Performance Metrics for KLAM-21 Runs')
                table.append(NoEscape(klam_stats.to_latex(index=True)))
        with doc.create(Subsection('Flood Fill Runs')):
            with doc.create(Table(position='h!')) as table:
                table.add_caption('Performance Metrics for Flood Fill Runs')
                table.append(NoEscape(ff_stats.to_latex(index=True)))
        with doc.create(Subsection('Statistical Comparison')):
            doc.append("A Mann-Whitney U test was performed to check for significant differences in performance metrics.")
            doc.append(NoEscape(r'\begin{verbatim}'))
            doc.append(stat_tests_results)
            doc.append(NoEscape(r'\end{verbatim}'))

    with doc.create(Section('Feature Space Analysis (Phenotypes)')):
        doc.append("Aggregate heatmaps show the average fitness found in each cell of the feature space across all 10 runs for the first two feature dimensions.")
        with doc.create(Figure(position='h!')) as fig:
            fig.add_image(figures['klam_heatmap'], width=NoEscape(r'0.8\textwidth'))
            fig.add_caption('Aggregate Archive Heatmap for KLAM-21 Runs')
        with doc.create(Figure(position='h!')) as fig:
            fig.add_image(figures['ff_heatmap'], width=NoEscape(r'0.8\textwidth'))
            fig.add_caption('Aggregate Archive Heatmap for Flood Fill Runs')
        doc.append(NoEscape(r'\newpage'))
        doc.append("The t-SNE plot projects the high-dimensional phenotypes into a 2D space to visualize structural similarities and differences.")
        with doc.create(Figure(position='h!')) as fig:
            fig.add_image(figures['pheno_tsne'], width=NoEscape(r'0.8\textwidth'))
            fig.add_caption('t-SNE projection of phenotypes, colored by evaluation method.')

    with doc.create(Section('Genetic Analysis')):
        doc.append("The t-SNE plot below projects the high-dimensional genomes into a 2D space to visualize diversity in the genetic search space.")
        with doc.create(Figure(position='h!')) as fig:
            fig.add_image(figures['geno_tsne'], width=NoEscape(r'0.8\textwidth'))
            fig.add_caption('t-SNE projection of genomes, colored by evaluation method.')
            
    try:
        report_path = os.path.join(REPORT_DIR, 'analysis_report')
        doc.generate_pdf(report_path, clean_tex=True)
        logging.info(f"Successfully generated PDF report: {report_path}.pdf")
    except Exception as e:
        logging.error(f"PDF Generation Failed. Error: {e}")
        logging.warning("Please ensure a LaTeX distribution (like MiKTeX, TeX Live, or MacTeX) is installed and that 'pdflatex' is in your system's PATH.")
        logging.info(f"The LaTeX source file was saved as: {report_path}.tex")

def main():
    """Main analysis workflow."""
    logging.info("--- Starting Analysis Script ---")
    
    klam_archive_paths = find_archive_paths(KLAM_DIR)
    ff_archive_paths = find_archive_paths(FF_DIR)
    
    if not klam_archive_paths or not ff_archive_paths:
        logging.error("Could not find archives in one or both directories. Aborting analysis.")
        return

    klam_data = extract_data_from_archive_paths(klam_archive_paths)
    ff_data = extract_data_from_archive_paths(ff_archive_paths)
    
    
    dataset_output_path = os.path.join(REPORT_DIR, 'klam_elites_dataset.npz')
    np.savez_compressed(
        dataset_output_path,
        genomes=klam_data['solutions'],
        features=klam_data['features'],
        phenotypes=klam_data['phenotypes'],
        objectives=klam_data['objectives']
    )
    logging.info(f"Saved complete KLAM elites dataset (genomes, features, phenotypes) to: {dataset_output_path}")

    klam_stats_summary = klam_data['stats_df'].agg(['mean', 'std']).round(2)
    ff_stats_summary = ff_data['stats_df'].agg(['mean', 'std']).round(2)
    stat_tests_results = perform_statistical_tests(klam_data['stats_df'], ff_data['stats_df'])

    figures = {}
    figures['klam_heatmap'] = os.path.join(REPORT_DIR, 'klam_aggregate_heatmap.png')
    generate_aggregate_heatmap(klam_archive_paths, "Aggregate Archive Heatmap (KLAM-21)", figures['klam_heatmap'])
    
    figures['ff_heatmap'] = os.path.join(REPORT_DIR, 'ff_aggregate_heatmap.png')
    generate_aggregate_heatmap(ff_archive_paths, "Aggregate Archive Heatmap (Flood Fill)", figures['ff_heatmap'])

    figures['pheno_tsne'] = os.path.join(REPORT_DIR, 'phenotype_tsne_projection.png')
    generate_tsne_plot(klam_data['phenotypes'], ff_data['phenotypes'], 'Phenotype', figures['pheno_tsne'])

    figures['geno_tsne'] = os.path.join(REPORT_DIR, 'genome_tsne_projection.png')
    generate_tsne_plot(klam_data['solutions'], ff_data['solutions'], 'Genome', figures['geno_tsne'])

    generate_latex_report(klam_stats_summary, ff_stats_summary, stat_tests_results, figures)
    logging.info("--- Analysis Script Finished ---")

if __name__ == "__main__":
    main()