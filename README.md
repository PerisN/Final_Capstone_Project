# Intelligent Insurance Claims Triage and Severity Prediction.

### 1. Project Description

This project focuses on developing an **intelligent insurance claims triage and severity prediction system**. The system will analyse information provided when a claim is reported, including the customer's details, accident information, work circumstances, initial cost estimate and the written description of the claim. It will then assess the likely severity of the claim and help determine which claims may require greater attention.

**Claims triage** means sorting and prioritising claims based on their level of urgency or seriousness. For example, a straightforward, low-cost claim could be processed normally, while a complex or potentially high-cost claim could be prioritised for more detailed assessment.

**Severity prediction** refers to estimating how costly or serious a claim is likely to become based on the information available when the claim is reported. For example, a claim involving a minor injury may require less attention than one involving a serious injury and potentially high compensation costs.

The overall goal is to help insurance companies process claims more efficiently, prioritise cases that require greater attention and support claims handlers in making faster and more consistent decisions.

### 2. Problem Statement

Insurance companies receive a large volume of claims that must be reviewed and processed efficiently. When this process is largely manual, it can lead to delays, inconsistent decisions and difficulties in identifying claims that require greater attention.

Claims contain written descriptions of incidents as well as information about the claimant and accident. Reviewing and interpreting this information manually can be time-consuming, particularly when insurers need to process many claims and determine which cases should be handled first.

This project addresses this problem by developing an **Intelligent Insurance Claims Triage and Severity Prediction System**. The system will analyse information provided when a claim is reported to estimate its likely outcome and help prioritise claims for processing. This can support claims handlers in making faster and more consistent decisions while allowing greater attention to be given to complex or potentially high-cost cases.

## 3. Project Aim

The aim of this project is to develop an intelligent system to help insurance companies **process claims more efficiently, prioritise resources, reduce delays and make more consistent decisions**, while allowing human claims handlers to focus their attention on the cases that need it most.

### 4. Research Questions

1. How accurately can structured insurance claim information be used to predict claim severity?

2. How accurately can information from claim descriptions be used to predict claim severity?

3. Does combining claim-description information with structured claim information improve claim severity prediction?

4. How can predicted claim severity be used to prioritise insurance claims for processing?

### 5. Study Objectives

The objectives of this project are to:

1. **To determine how accurately structured insurance claim information can be used to predict claim severity.**

2. **To determine how accurately information from claim descriptions can be used to predict claim severity.**

3. **To determine whether combining claim-description information with structured claim information improves claim severity prediction.**

4. **To determine how predicted claim severity can be used to prioritise insurance claims for processing.**

### 6. Dataset Description

The project will use the **DataRobot Insurance Claims Triage dataset**: https://s3.amazonaws.com/datarobot-doc-assets/DR_Demo_Statistical_Case_Estimates.csv, which contains information about insurance claims and the circumstances surrounding each claim.

The dataset includes:
- **Claim description** – a written description of the incident.
- **Claimant information** – such as age, gender, marital status and dependants.
- **Employment information** – such as working hours and employment type.
- **Accident information** – such as the accident date, accident hour and reporting delay.
- **Financial information** – including the initial case estimate and incurred claim amount.

### 7. Dataset Characteristics

| Characteristic | Details |
|---|---|
| Dataset | DataRobot Insurance Claims Triage Dataset |
| Number of rows | 21692 |
| Number of columns | 16 |
| Data types | Numerical, categorical, date/time and text |
| Target variable | Incurred |
| Text variable | ClaimDescription |

### 8. Project Scope

The project will cover:

- Analysis and preprocessing of structured insurance claim information.
- Analysis of written claim descriptions.
- Prediction of claim severity using information available at the time of reporting.
- Investigation of whether combining textual and structured information improves prediction.
- Development of a prioritisation approach based on predicted claim severity.
- Evaluation of the performance and reliability of the developed system.

The project will focus on supporting claims handlers rather than replacing human decision-making. The system will provide predictions and prioritisation information to assist with the claims-handling process.

### 9. Methodology

The project will follow these main stages:

1. **Exploratory Data Analysis**  
   Examine the dataset, understand the variables, identify patterns, and investigate the distribution of claim outcomes.

2. **Data Preprocessing**  
   Clean the data, handle missing values and inconsistencies, and prepare the structured and textual information for analysis.

3. **Text Analysis**  
   Analyse the written claim descriptions to identify useful information that can contribute to predicting claim severity.

4. **Feature Integration**  
   Combine information from the claim descriptions with relevant structured claim variables.

5. **Severity Prediction**  
   Develop models to predict the likely severity of an insurance claim.

6. **Claim Prioritisation**  
   Use the predicted severity to assign claims different levels of priority for further processing.

7. **Model Evaluation**  
   Evaluate how accurately the system predicts claim severity and assess its usefulness for claim prioritisation.

### 10. Evaluation

The developed system will be evaluated based on its ability to accurately predict insurance claim severity and support claim prioritisation.

The evaluation will consider:

- **Prediction accuracy** – how well the model predicts claim severity.
- **Error analysis** – the extent and type of prediction errors made by the model.
- **Comparison of approaches** – whether combining claim descriptions with structured information improves performance.
- **Prioritisation performance** – whether the predicted severity can effectively distinguish claims requiring different levels of attention.
- **Model interpretability** – understanding which information contributes most to the predictions.

### 11. Tools and Technologies

- **Python** – for data analysis, preprocessing, modelling and evaluation.

- **Pandas & NumPy** – for data manipulation and numerical analysis.

- **Scikit-learn** – for preprocessing, dimensionality reduction, machine learning, and model evaluation.

- **Matplotlib & Seaborn** – for exploratory data analysis and visualisation.

- **Natural Language Processing (BERT)** – converts the written `ClaimDescription` into numerical representations that capture meaningful information from the claim narrative.

- **Dimensionality Reduction (PCA)** – reduces the number of features while retaining important information, potentially making modelling more manageable.

- **Manifold Learning (UMAP, t-SNE)** – visualises patterns and relationships between claims in a lower-dimensional space to help explore similarities between claims.

- **Clustering (K-Means, GMM, DBSCAN)** – groups claims based on similarities in their characteristics, helping investigate different claim profiles and patterns.

- **Machine Learning (Random Forest, Gradient Boosting)** – predicts the likely financial cost of claims using structured information and text-based features.

- **Deep Learning (Neural Network)** – provides an alternative approach to claim-cost prediction, allowing its performance to be compared with traditional machine learning models.

- **Model Explainability (SHAP)** – identifies which claim characteristics contribute to severity predictions, helping explain the model's decisions.

- **Model Evaluation (MAE, RMSE, R²)** – measures and compares how accurately the models predict claim costs.

### 12. Project Workflow

**Insurance Claim Data**  
↓  
**NLP (BERT or FinBERT)** – Extract meaningful information from claim descriptions  
↓  
**Feature Integration** – Combine text-based and structured claim information  
↓  
**Dimensionality Reduction & Manifold Learning** – Reduce and visualise the feature space  
↓  
**Clustering & Anomaly Detection** – Identify claim patterns and unusual claims  
↓  
**Severity Prediction** – Predict the likely severity of each claim  
↓  
**Claim Prioritisation** – Use predicted severity to assign appropriate priority levels  
↓  
**Model Explainability (SHAP)** – Identify the factors influencing predictions

### 13. Expected Outcomes

The project is expected to achieve the following outcomes:

1. **Structured Information:** Determine how accurately structured insurance claim information can predict claim severity.

2. **Claim Descriptions:** Determine how accurately information from claim descriptions can predict claim severity.

3. **Combined Information:** Establish whether combining claim descriptions with structured claim information improves severity prediction.

4. **Claim Prioritisation:** Develop a prioritisation approach that uses predicted claim severity to identify claims requiring different levels of attention.