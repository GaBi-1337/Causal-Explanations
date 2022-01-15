# Explaining The Decisions of Black-Box Classifiers by Aggregating Causes
This repository contains code for the paper "Model Explanations via the Axiomatic Causal Lens"

## Usage
To explain the decision of a point of interest by a black-box model:
  - Import the _Causal_Explanations_ and the _Mapper_ classes from explain.py
  - The class _Mapper_ maps categorical values to One Hot Encoded (OHE) values. This is required by _Causal_Explanations_ to translate between OHE values and their corresponding categorical values while computing the value function. The class _Mapper_ requires only the dataset as the input. For correct explanations it is ideal to train a black-box model with the dataset returned from the get_data method (if other OHE methods are used, such as the one provided by sklearn, the columns get rearranged).
  - The class _Causal_Explanations_ needs the black-box model, the point of interest, an object of type mapper, and a list of baselines that have the opposite outcome. 
  - A power index of choice needs to be called to get the explanations. Moreover, each power index has a sampling version as well. These sampling versions need an additional two parameters ε and δ 
