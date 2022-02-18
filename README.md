# Explaining The Decisions of Black-Box Classifiers by Aggregating Causes
This repository contains code for the paper "Model Explanations via the Axiomatic Causal Lens"

## Usage
To explain the decision of a point of interest by a black-box model:
  - Import the _Causal_Explanations_ and the _Mapper_ classes from explain.py
  - The class _Mapper_ maps categorical values to One Hot Encoded (OHE) values. This is required by _Causal_Explanations_ to translate between OHE values and their corresponding categorical values while computing the value function. The class _Mapper_ requires only the dataset as the input. For correct explanations it is ideal to train a black-box model with the dataset returned from the get_data method (if other OHE methods are used, such as the one provided by sklearn, the columns get rearranged).
  - The class _Causal_Explanations_ needs the black-box model, the point of interest (as it is as opposed to OHE version), an object of type mapper, and a list of baselines that have the opposite outcome (as they are as opposed to OHE version). 
  - A power index of choice needs to be called to get the explanations. Moreover, each power index has a sampling version as well. These sampling versions need an additional two parameters: ε and δ. Where ε is the deviation of the approximated computations from the actual and δ is the probability that the approximated computation is outside of this deviation. These two parameters are used to calculate the required number of samples. 

An example usage can be seen in the example.py file.

## How it works
Once the power index is called, a subset S is then picked from the power set of the features (in the case of exact computations) or from the randomly sampled sets (in the case of sampling versions). This set S is used to pick the features from the list of baselines that have the opposite outcome. The value function is then used to determine the critical features of the set S; a feature is critical in S if removing that feature from S causes the value function to change i.e. a feature *i* is critical if *|v(S) - v(S \ i) |= 1*.

### Indices that Aggregate Minimal Causes
A set of causes S is minimal if all features in the set are critical

  - Responsibility index: For this index a feature’s importance is higher if the minimal cause the feature appears is smaller.

  - Holler-Packel index: For this index a feature’s importance is higher if a feature appears in more minimal causes.

  - Deegan-Packel index: For this index a feature’s importance is determined by both the size and number of minimal causes the feature appears in.

### Indices that Aggregate Minimal Causes
A set of causes S is quasi-minimal if at least one feature is critical.

  - Shapley-Shubik index: For this index each feature that is critical in S is assigned a weight of *(|S| - 1)(n - |S|) / n!*

  - Banzhaf index: For this index a feature’s importance is higher if a feature appears in more quasi-minimal causes and is critical in that set.

  - Johnston index: For this index a feature’s importance is higher if the quasi-minimal cause the feature appears is smaller.
