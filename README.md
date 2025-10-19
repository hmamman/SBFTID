# Search-Based Fairness Testing for Detecting Individual Discrimination in Machine Learning-Based Software
## Thesis Abstract

Machine learning-based software (MLS) are increasingly used for autonomous decisions in critical domains such as healthcare, finance, hiring, education, and criminal justice. However, MLS produces unfair, discriminatory decisions against individuals based on protected attributes such as race, gender, and age. Extensive fairness testing is essential to detect discrimination before deploying MLS. Individual Fairness Testing (IFT) detects discrimination when two individuals who differ in protected attributes receive different outcomes from MLS. Data instances that reveal such discrimination are called discriminatory instances. Search-based fairness testing (SBFT) has received considerable attention for its ability to efficiently explore vast input spaces to detect discriminatory instances. Despite recent advancements, existing SBFT approaches demonstrate significant limitations, including low effectiveness, poor efficiency, and limited scalability. To address these changes, this study proposes new search-based frameworks for IFT. 

## Proposed Frameworks
1. **FairES:**  Modified Evolution Strategy for Individual Fairness Testing
2. **FairESTEO:**  Hybrid Evolution Strategy with Thermal Exchange Optimization for Individual Fairness Testing
3. **FairPHS:** Individual Fairness Testing through a Memory-Guided Probabilistic Hybrid Search
4. **FairSIFT:** Scalable Individual Fairness Testing via Batch Inference


## Installation

1. Download/Clone the repository:
   ```bash
   Download from: https://github.com/hmamman/SBFTID/archive/refs/heads/main.zip
   Unzip and cd into the directory
   OR
   Clone from: https://github.com/hmamman/SBFTID


2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
## Datasets and Protected Attributes
| Dataset | Protected Attribute | Index (Starts at 1) |
|---------|---------------------|---------------------|
| census  | sex                 | 9                   |
|         | age                 | 1                   |
|         | race                | 8                   |
| credit  | sex                 | 9                   |
|         | age                 | 13                  |
| bank    | age                 | 1                   |
|         | marital             | 3                   |
| compas  | sex                 | 1                   |
|         | age                 | 2                   |
|         | race                | 3                   |
| meps    | sex                 | 3                   |

Dataset and protected attribute names are case-sensitive.

## Running Experiment

### Command-Line Arguments

The script accepts the following arguments:

- `--dataset_name`: (string) Name of the dataset to use in the experiment. The default is `'census'`.
  - Example: `--dataset_name census`

- `--sensitive_name`: (string) Name of the protected attribute for fairness testing (e.g., `sex`, `age`, `race`). The default is `'age'`.
  - Example: `--sensitive_name sex`

- `--classifier_name`: (string) Name of the classifier to use (e.g., `mlp`, `dt`, `rf`, ect.). The default is `'dt'`.
  - Example: `--classifier_name svm`

- `--max_allowed_time`: (integer) Maximum time in seconds for the experiment to run. The default is `3600` seconds (1 hour).
  - Example: `--max_allowed_time 3600`

### Example Usage

To run the any framework (e.g., fairses, fairpes, fairces, fairesteo, fairphs, fairsift):
```bash
python ./tutorial/fairsift.py --classifier_name dt --dataset_name census --sensitive_name age --max_allowed_time 3600
```

To run a specific baseline approach included in this repository (e.g., aequitas, sg, adf, neuronfair, expga, vbtx, aft):
```bash
python ./baseline/expga/expga.py --classifier_name dt --dataset_name census --sensitive_name age --max_allowed_time 3600
```

You can also run all experiments for an approach by running the main.py file:
```bash
python ./main.py --approach_name fairsift --max_allowed_time 3600 --max_iteration 1  
```
 
## IFT Baseline Methods

The table below summarizes key details of each baseline methods used to evaluate the proposed frameworks.

| Method   | Paper Title                                                                                           | GitHub Repo                                                        | Name     |
|----------|--------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------|----------|
| ADF      | *Automatic Fairness Testing of Neural Classifiers through Adversarial Sampling*                       | [pxzhang94/ADF](https://github.com/pxzhang94/ADF)                   | adf      |
| NeuronFair | *Interpretable White-Box Fairness Testing through Biased Neuron Identification*                     | [haibinzheng/NeuronFair](https://github.com/haibinzheng/NeuronFair) | neuronfair |
| AEQUITAS | *Automated Directed Fairness Testing*                                                                  | [sakshiudeshi/Aequitas](https://github.com/sakshiudeshi/Aequitas)  | aequitas |
| SG       | *Black-Box Fairness Testing of Machine Learning Models*                                                | [waving7799/ExpGA](https://github.com/waving7799/ExpGA) (SG module) | sg       |
| ExpGA    | *Explanation-Guided Fairness Testing Through Genetic Algorithm*                                        | [waving7799/ExpGA](https://github.com/waving7799/ExpGA)             | expga    |
| SBFT     | *Search-Based Fairness Testing for Regression-Based Machine Learning Systems*                          | [search-based-fairness-testing/sbft](https://github.com/search-based-fairness-testing/sbft) | sbft     |
| AFT      | *Approximation-guided Fairness Testing through Discriminatory Space Analysis*                          | [toda-lab/AFT](https://github.com/toda-lab/AFT)                     | aft  |
| Vbt-X    | *Verification-Based Testing with Constraint Solving* (supporting method cited by AFT)                  | [toda-lab/Vbt-X](https://github.com/toda-lab/Vbt-X)                 | vbtx    |

## Notes
- **ADF** and **NeuronFair** are white-box fairness testing methods that use internal model structures (e.g., gradients and neurons).
- **AEQUITAS**, **SG**, **ExpGA**, and **SBFT** are black-box testing tools using probabilistic search, symbolic execution, and evolutionary algorithms.
- **AFT** and **Vbt-X** apply verification-based techniques, with AFT using approximation strategies to reduce computational overhead.
- The SG implementation was sourced from the ExpGA repository.

