# Harnessing Cell Painting and Machine Learning to Predict Drug Safety and Pharmacology  
### Raquel Parrondo, Albert A. Antolin, and Jordi Mestres

Here it is available the code (/src) and different notebooks (/notebooks) to reproduce our work and analyse the results obtained in the study: **Harnessing Cell Painting and Machine Learning to Predict Drug Saftey and Phramacology**. 

- **[data](data/)** contains all predictive model results to evaluate the outcomes and visualize the results.
    - **[3_data](data/3_data/)** contains predictive performeance of the individual regressors predicting morphological features based on chemical data.
    - **[4_data](data/4_data/)** contains predictive performance of the individual models predicting target bioactivity based on physicochemical, structural and morphological descriptors. 
    - **[5_data](data/5_data/)** contains predictive performance of the individual classifiers predicting side-effect presence based on physicochemical, structural and morphological descriptors.
    - **[cp_top_models](data/cp_top_models/)** contains the six top-performing pharmacology models: CDK5, MITF, RET, MAP4K5, MAP2K1, MINK1.
- **[src](src/)** contains the source code cotaining all functions called by the Notebooks.
- **[notebooks](notebooks/)** contains the necessary Notebooks to reproduce our work and inspect the results.
    - **1_CellPaintingData_prepocessing.ipynb** includes the pipeline implemented on the project to preprocess the compound morphological profiles derived from the cell painting assay.
    - **2_Molecular_Descriptors_Calculation.ipynb** calculates the four molecular descriptors considered in the project to represent the chemical properties and structural information of compounds.
    - **3_CellPaintingFeatures_PredictiveModels.ipynb** trains and evaluates the individual regressors that predict cell painting morphological features.
    - **4_Pharmacology_PredictiveModels.ipynb** trains and evaluates the invidual models predicting pharmacological bioactivity.
    - **5_Safety_PredictiveModels.ipynb** trains and evaluates the individual classifiers predicting side-effect reportance.
  
*Important*: Data folder contains predictive model results to analyse the outcomes. Original molecule data to train the models will be made accessible upon request.
