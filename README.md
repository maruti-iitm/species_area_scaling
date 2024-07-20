# "Quantifying DOM Scaling Relationships and Trends in Watersheds"

Welcome to the Species Richness Scaling Law repository! This research project explores the relationship between species richness (i.e., compound metrics) and watershed characteristics using various environmental datasets and data-driven methods. The repository includes data and associated links, notes in curating the data (e.g., graphics, slides), and data analysis scripts in understanding species-watershed relationships.

## Research overview

Our aim is to analyze how species richness scales with the habitat features. This relationship is crucial for biodiversity conservation, ecological studies, and environmental management. By leveraging multiple environmental datasets and advanced statistical methods, the project provides insights into the species-area relationship across different regions and ecosystems.

### Conceptual diagram:

![Concept](https://github.com/maruti-iitm/species_area_scaling/blob/main/Notes/Graphics/EMSL_0555_WHONDRS_TOC-03.png)


### Workflow diagram:
![Workflow](https://github.com/maruti-iitm/species_area_scaling/blob/main/Notes/Graphics/EMSL_0555_WHONDRS_TOC-02.png)






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

## Data availability
Additional information, including summarized data files, metadata, and data from the figures and supplementary information that is not available in this GitHub repository, can be accessed publicly in an open Zenodo data repository at the following link: XXXX

## Acknowledgements
We acknowledge the work of the WHONDRS team from Pacific Northwest National Laboratory for their effort developing the WHONDRS program and providing the datasets. MKM’s research was supported by the Environmental Molecular Sciences Laboratory, a DOE Office of Science User Facility sponsored by the Biological and Environmental Research program under Contract No. DE-AC05-76RL01830 (Award DOIs: 10.46936/intm.proj.2022.60592/60008643; 10.46936/intm.proj.2023.60904/60008965). MN’s research was supported by the Watershed Function Science Focus Area funded by the U.S. Department of Energy, Office of Science, Office of Biological and Environmental Research under Award Number DE-AC02-05CH1123. The authors acknowledge the contributions of Nathan Johnson at PNNL’s Creative Services, who developed the conceptual graphics in this paper.

PNNL-SA-XXXXX

## Competing interests
The author(s) declare no competing interests 

## Author contributions
All authors contributed to visioning, writing, analysis, data interpretation, and concept development. All authors fulfilled the CREDIT criteria to warrant authorship.

## Disclaimer
This research work was prepared as an account of work sponsored by an agency of the United States Government. Neither the United States Government nor any agency thereof, nor any of their employees, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness of any information, apparatus, product, or process disclosed, or represents that its use would not infringe privately owned rights. Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.