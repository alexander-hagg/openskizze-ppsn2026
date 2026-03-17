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
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import yaml
import logging

# PyLaTeX for report generation
from pylatex import Document, Section, Figure, Table, Subsection, NoEscape

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_KLAM_DIR = "results/sail_klam_training"
DEFAULT_REPORT_DIR = "results/analysis_report"


def find_archive_paths(directory: str) -> list:
    """Finds all final QD archive paths from a results directory."""
    logging.info(f"Searching for archives in: {directory}")
    
    # Try multiple patterns to handle different directory structures
    patterns = [
        f"{directory}/**/FinalQD_archive.pkl",      # Nested anywhere
        f"{directory}/**/*FinalQD_archive.pkl",     # With prefix, nested
        f"{directory}/*FinalQD_archive.pkl",        # Direct in directory
        f"{directory}/rep_*/**/*FinalQD_archive.pkl", # Standard rep_X structure
    ]
    
    archive_paths = []
    for pattern in patterns:
        found = glob.glob(pattern, recursive=True)
        archive_paths.extend(found)
    
    # Remove duplicates while preserving order
    archive_paths = list(dict.fromkeys(archive_paths))
    
    logging.info(f"Found {len(archive_paths)} archive paths.")
    for path in archive_paths:
        logging.info(f"  - {path}")
    
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
        'objectives': np.concatenate(all_objectives) if all_objectives else np.array([]),
        'stats_df': pd.DataFrame(stats_list)
    }

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

def generate_latex_report(klam_stats, figures, output_dir):
    """Generates a PDF report summarizing the analysis."""
    logging.info("Generating LaTeX report...")
    doc = Document(geometry_options={"tmargin": "1in", "lmargin": "1in"})
    doc.preamble.append(NoEscape(r'\usepackage{graphicx}\usepackage{booktabs}\usepackage{verbatim}'))
    doc.append(NoEscape(r'\title{Analysis of SAIL Optimization Runs (KLAM-21)}\author{Automated Analysis Script}\date{\today}\maketitle'))
    
    with doc.create(Section('Performance Summary')):
        doc.append("This section shows performance metrics (QD Score, Coverage) across all replicate runs for the KLAM-21 method.")
        with doc.create(Table(position='h!')) as table:
            table.add_caption('Performance Metrics for KLAM-21 Runs')
            table.append(NoEscape(klam_stats.to_latex(index=True)))

    with doc.create(Section('Feature Space Analysis')):
        doc.append("The aggregate heatmap shows the average fitness found in each cell of the feature space across all runs for the first two feature dimensions.")
        if 'klam_heatmap' in figures and os.path.exists(figures['klam_heatmap']):
            with doc.create(Figure(position='h!')) as fig:
                fig.add_image(figures['klam_heatmap'], width=NoEscape(r'0.8\textwidth'))
                fig.add_caption('Aggregate Archive Heatmap for KLAM-21 Runs')
            
    try:
        report_path = os.path.join(output_dir, 'analysis_report')
        doc.generate_pdf(report_path, clean_tex=True)
        logging.info(f"Successfully generated PDF report: {report_path}.pdf")
    except Exception as e:
        logging.error(f"PDF Generation Failed. Error: {e}")
        logging.warning("Please ensure a LaTeX distribution (like MiKTeX, TeX Live, or MacTeX) is installed and that 'pdflatex' is in your system's PATH.")
        logging.info(f"The LaTeX source file was saved as: {report_path}.tex")


def main():
    """Main analysis workflow."""
    parser = argparse.ArgumentParser(
        description="Analyze SAIL optimization results and export solutions to NPZ format."
    )
    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        default=DEFAULT_KLAM_DIR,
        help=f"Directory containing SAIL results (default: {DEFAULT_KLAM_DIR})"
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=DEFAULT_REPORT_DIR,
        help=f"Directory for output files (default: {DEFAULT_REPORT_DIR})"
    )
    parser.add_argument(
        '--output-name',
        type=str,
        default='klam_elites_dataset',
        help="Base name for output NPZ file (default: klam_elites_dataset)"
    )
    parser.add_argument(
        '--no-report',
        action='store_true',
        help="Skip LaTeX report generation"
    )
    parser.add_argument(
        '--no-heatmap',
        action='store_true',
        help="Skip heatmap generation"
    )
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.info("--- Starting Analysis Script ---")
    logging.info(f"Input directory: {args.input_dir}")
    logging.info(f"Output directory: {args.output_dir}")
    
    klam_archive_paths = find_archive_paths(args.input_dir)
    
    if not klam_archive_paths:
        logging.error(f"Could not find archives in {args.input_dir}. Aborting analysis.")
        return

    klam_data = extract_data_from_archive_paths(klam_archive_paths)    
    
    # Export to NPZ
    dataset_output_path = os.path.join(args.output_dir, f'{args.output_name}.npz')
    np.savez_compressed(
        dataset_output_path,
        genomes=klam_data['solutions'],
        features=klam_data['features'],
        phenotypes=klam_data['phenotypes'],
        objectives=klam_data['objectives']
    )
    logging.info(f"Saved complete KLAM elites dataset to: {dataset_output_path}")
    logging.info(f"  - Genomes shape: {klam_data['solutions'].shape}")
    logging.info(f"  - Features shape: {klam_data['features'].shape}")
    logging.info(f"  - Phenotypes shape: {klam_data['phenotypes'].shape}")
    logging.info(f"  - Objectives shape: {klam_data['objectives'].shape}")

    klam_stats_summary = klam_data['stats_df'].agg(['mean', 'std']).round(2)
    
    figures = {}
    
    if not args.no_heatmap:
        figures['klam_heatmap'] = os.path.join(args.output_dir, 'klam_aggregate_heatmap.png')
        generate_aggregate_heatmap(klam_archive_paths, "Aggregate Archive Heatmap (KLAM-21)", figures['klam_heatmap'])
    
    if not args.no_report:
        generate_latex_report(klam_stats_summary, figures, args.output_dir)
    
    logging.info("--- Analysis Script Finished ---")

if __name__ == "__main__":
    main()