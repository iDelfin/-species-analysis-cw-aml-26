```text
│   .gitattributes
│   .gitignore
│   preprocessing.py
│   README.md
│   requirements.txt|

├───climate
│   │   bioclim_h3_res4.npz             # Created by preprocessing.py
│   │   bioclim_norm_stats.npz          # Created by preprocessing.py
│   │   future_bioclim_h3_res4.npz      # Created by preprocessing.py
│   │   preprocessed_cells_bio.npz      # Created by preprocessing.py
│   │   preprocessed_species_bio.npz    # Created by preprocessing.py
│   │   preprocessed_test_bio.npz       # Created by preprocessing.py
│   │   preprocessed_test_future.npz    # Created by preprocessing.py
│   │
│   ├───future_wc2.1_10m_bio
│   │       wc2.1_10m_bioc_MPI-ESM1-2-HR_ssp245_2041-2060.tif
│   │
│   └───wc2.1_10m_bio
│           wc2.1_10m_bio_1.tif
│           wc2.1_10m_bio_10.tif
│           wc2.1_10m_bio_11.tif
│           wc2.1_10m_bio_12.tif
│           wc2.1_10m_bio_13.tif
│           wc2.1_10m_bio_14.tif
│           wc2.1_10m_bio_15.tif
│           wc2.1_10m_bio_16.tif
│           wc2.1_10m_bio_17.tif
│           wc2.1_10m_bio_18.tif
│           wc2.1_10m_bio_19.tif
│           wc2.1_10m_bio_2.tif
│           wc2.1_10m_bio_3.tif
│           wc2.1_10m_bio_4.tif
│           wc2.1_10m_bio_5.tif
│           wc2.1_10m_bio_6.tif
│           wc2.1_10m_bio_7.tif
│           wc2.1_10m_bio_8.tif
│           wc2.1_10m_bio_9.tif
│           wc2.1_10m_elev.tif
│
├───report
│   │   lineno.sty
│   │   nicefrac.sty
│   │   nips_2016.sty
│   │   refs.bib
│   │   template.tex
│   │
│   ├───Data Exploration Graphs
│   │       distribution_of_observations.png
│   │       observation_species_count_bar_chart.pdf
│   │
│   └───images
│           auc_predict_species.png
│           bio_app.png
│           coord_app.png
│           cror_turt_map_heatmap.png
│           current.pdf
│           future.pdf
│           Gantt_CW_AML.png
│           learningcurve_acc.png
│           learningcurve_auc.png
│           learningcurve_bce.png
│           learningcurve_map.png
│           output_features.png
│           per_class_map_heatmap.png
│           per_class_map_relchange.png
│
└───species
        preprocessed_cells.npz          # Created by preprocessing.py
        preprocessed_cells_test.npz     # Created by preprocessing.py
        preprocessed_species.npz        # Created by preprocessing.py
        preprocessed_species_test.npz   # Created by preprocessing.py
        species_taxonomy.csv            # Created by preprocessing.py
        species_test.npz
        species_train.npz
        species_train_extra.npz
        
