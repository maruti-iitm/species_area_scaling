# Research: Quantifying Dissolved Organic Matter Scaling Relationships and Trends in Watersheds

Welcome to the Species Richness Scaling Law repository! This research project explores the relationship between species richness (i.e., compound metrics) and watershed characteristics using various environmental datasets and data-driven methods. The repository includes data and associated links, notes in curating the data (e.g., graphics, slides), and data analysis scripts in understanding species-watershed relationships.

## Research overview

Our aim is to analyze how species richness scales with the habitat features. This relationship is crucial for biodiversity conservation, ecological studies, and environmental management. By leveraging multiple environmental datasets and advanced statistical methods, the project provides insights into the species-area relationship across different regions and ecosystems.

### Conceptual diagram:

![Concept](https://github.com/maruti-iitm/species_area_scaling/blob/main/Notes/Graphics/EMSL_0555_WHONDRS_TOC-03.png)


### Workflow diagram:
![Workflow](https://github.com/maruti-iitm/species_area_scaling/blob/main/Notes/Graphics/EMSL_0555_WHONDRS_TOC-02.png)

### Repo contents:

- **Data processing**: Scripts for preprocessing and cleaning various environmental datasets.
- **Analysis methods**: Implementation of data-driven and feature importance models to assess species-metric-scaling relationships.
- **Visualization**: Graphical representations of the analysis results usign `matplotlib`.
- **Code documentation**: Detailed doc strings to help users replicate and extend the analyses.
- **Preliminary Google Colab notebook**: https://colab.research.google.com/drive/10wx9QCxRWWmY2Xi4WMAR-YZgvYbL_sty#scrollTo=0RGUt-SrhGUc

### Datasets

The research uses several datasets, including:

- **WHONDRS**: Data attribures such as water temperature, dissolved oxygen, and pH measurements.
- **StreamStats**: Hydrological and geographical data.
- **HydroSheds**: Hydrological data on river networks and watersheds.
- **EPAWaters**: Environmental Protection Agency catchment and watershed characteristics.

Refer to the `Data/` and `v1/Inputs_Outputs_v4/` directories.

### Methods

The research employs various statistical and machine learning methods to analyze species-richness-scaling relationships, including:

- F-test
- Mutual Information (MI)
- Random Forest (RF)
- SHAPley values
- Pearson and Spearman correlations

Each method's implementation is documented in the `v1/*.py` directory.

### Results

Results of the analyses are stored in the `v1/Plots_*/` and `v1/PCA_all/` directories, including:

- Feature importance scores
- Model performance metrics
- Visualizations of species-richness-scaling relationships

Refer to the `Slides/README.md` file for a summary of the findings.

## Contributing

To get started with the research, clone the repository and install the required dependencies. 
All the Python scripts are in `v1/get_*.py`

```
git clone https://github.com/maruti-iitm/species_area_scaling.git
cd species_area_scaling
pip install -r requirements.txt
```

We welcome contributions from the community! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-user-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-user-branch`).
6. Open a pull request.

**Contact** 

If you have questions or need help getting started, contact Maruti Mudunuru (<maruti@pnnl.gov>).

**Copyright guidelines**

To maintain compatibility with the project's overall licensing, please adhere to the following:

* **Datasets:** Do not include datasets with restricted licenses that prohibit free use or modification, as they may conflict with the project's MIT license.
* **Code Snippets:** Avoid using Python/R code snippets with restricted licenses that prevent free use or modification.

### Directory structure for contribution

```
tree -L 2 . >> tree.txt
```

```
.
├── Data
│             ├── AminoSugar.csv
│             ├── Carb.csv
│             ├── ConHC.csv
│             ├── Lignin.csv
│             ├── Lipid.csv
│             ├── Other.csv
│             ├── Processed_S19S_Sediments_Water_2-2_newcode.csv
│             ├── Protein.csv
│             ├── README.md
│             ├── Tannin.csv
│             ├── UnsatHC.csv
│             └── WHONDRS_S19S_Metadata_v3.csv
├── LICENSE
├── Notes
│             ├── Graphics
│             ├── Maruti_Working_Code.zip
│             ├── Supplementary_Material
│             ├── Topic-5_v1.docx
│             └── Topic5_Transformations-main.zip
├── README.md
├── Slides
│             ├── README.md
│             ├── hydrosheds_vs_SR_v10.pptx
│             ├── hydrosheds_vs_SR_v11.pptx
│             ├── hydrosheds_vs_SR_v12.pptx
│             ├── hydrosheds_vs_SR_v13.pptx
│             ├── hydrosheds_vs_SR_v14.pptx
│             ├── hydrosheds_vs_SR_v15.pptx
│             ├── hydrosheds_vs_SR_v16.pptx
│             ├── hydrosheds_vs_SR_v17.pptx
│             ├── hydrosheds_vs_SR_v18.pptx
│             ├── hydrosheds_vs_SR_v19.pptx
│             ├── hydrosheds_vs_SR_v3.pptx
│             ├── hydrosheds_vs_SR_v4.pptx
│             ├── hydrosheds_vs_SR_v5.pptx
│             ├── hydrosheds_vs_SR_v6.pptx
│             ├── hydrosheds_vs_SR_v7.pptx
│             ├── hydrosheds_vs_SR_v8.pptx
│             ├── hydrosheds_vs_SR_v9.pptx
├── requirements.txt
└── v1
    ├── 1_ATLAS_Docs
    ├── 2_EPAWaters_Docs
    ├── 3_WHONDRS_Docs
    ├── CQ_all
    ├── Important features identified in EPA-ACC adn ACW.docx
    ├── Inputs_Outputs_v4
    ├── PCA_all
    ├── Plots_EPAWaters_ACC
    ├── Plots_EPAWaters_ACW
    ├── Plots_HYDROSHEDS
    ├── Plots_StreamStats
    ├── Plots_WHONDRS
    ├── get_cq.py
    ├── get_ftrimp_acc.py
    ├── get_ftrimp_acw.py
    ├── get_ftrimp_hydrosheds.py
    ├── get_ftrimp_streamstats.py
    ├── get_ftrimp_whondrs.py
    ├── get_pca_analysis.py
    ├── get_pca_analysis_impftrs.py
    ├── get_plots_acc.py
    ├── get_plots_acw.py
    ├── get_plots_hydrosheds.py
    ├── get_plots_streamstats.py
    ├── get_plots_whondrs.py
    ├── get_processed_data_v3.py
    ├── get_sl_acc.py
    ├── get_sl_acw.py
    ├── get_sl_hydrosheds.py
    ├── get_sl_streamstats.py
    ├── get_sl_whondrs.py
    └── p_value_important_features.xlsx

18 directories, 59 files
```

```
pygount --format=summary .
```

```
|------------------------------------------------------------------|
| Language     |   Files |   % |   Code |   % |   Comment |   % |
|--------------|--------:|----:|-------:|----:|---------:|----:|
| Python       |      19 |  2.0 |   4177 | 64.2 |     1948 | 29.9 |
| Text only    |       6 |  0.6 |    109 | 95.6 |        0 |  0.0 |
| Markdown     |       4 |  0.4 |     82 | 38.5 |        0 |  0.0 |
| __unknown__  |     438 | 46.0 |      0 |  0.0 |        0 |  0.0 |
| __binary__   |     485 | 50.9 |      0 |  0.0 |        0 |  0.0 |
-------------------------------------------------------------------
| **Sum**      |     952 |100.0 |   4368 | 63.9 |     1948 | 28.5 |
-------------------------------------------------------------------
```

## Data availability
Additional information, including summarized data files, metadata, and data from the figures and supplementary information that is not available in this GitHub repository, can be accessed publicly in an open Zenodo data repository at the following link: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12789205.svg)](https://doi.org/10.5281/zenodo.12789205)

## Acknowledgements
We acknowledge the work of the WHONDRS team from Pacific Northwest National Laboratory for their effort developing the WHONDRS program and providing the datasets. MKM’s research was supported by the Environmental Molecular Sciences Laboratory, a DOE Office of Science User Facility sponsored by the Biological and Environmental Research program under Contract No. DE-AC05-76RL01830 (Award DOIs: 10.46936/intm.proj.2022.60592/60008643; 10.46936/intm.proj.2023.60904/60008965). MN’s research was supported by the Watershed Function Science Focus Area funded by the U.S. Department of Energy, Office of Science, Office of Biological and Environmental Research under Award Number DE-AC02-05CH1123. The authors acknowledge the contributions of Nathan Johnson at PNNL’s Creative Services, who developed the conceptual graphics in this paper.

PNNL-SA-XXXXX

## Competing interests
The author(s) declare no competing interests 

## Author contributions
All authors contributed to visioning, writing, analysis, data interpretation, and concept development. All authors fulfilled the CREDIT criteria to warrant authorship.

## Disclaimer
This research work was prepared as an account of work sponsored by an agency of the United States Government. Neither the United States Government nor any agency thereof, nor any of their employees, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness of any information, apparatus, product, or process disclosed, or represents that its use would not infringe privately owned rights. Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.