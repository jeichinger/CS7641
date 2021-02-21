For Assignment 1 analysis I utilized sklearn. A lot of the code I used can be found in the skearn documentation. I've highlighted the URLs used in this assignment here:
- https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py - utilized to plot Cost Complexity Pruning
- https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py - utilized to tune hyperparameter
- Oreilly Hands-On Machine Learning with Python, Keras, and Tensorflow, specifically chapter 2 - adapted for generating learning curves.
- Sklearn source code for making confusion matricies. Copied and adapted to make matricies look neater/ fit in report.

The original datasets can be found at these links:
https://www.kaggle.com/mullerismail/richters-predictor-modeling-earthquake-damage
https://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set

The zip file containing my code can be pulled from the following link: https://github.com/jeichinger/CS7641/blob/master/AS1.zip

The modified datasets(Described in report) can be found in the Data folder.

To assist in running the experiment code, I've created a yml file located in the AS1 root directory. Simply create a virtual envirionment from the file, activate it, and run the Experiment.py file.

To remove the potential of OS issues making directories, I've included the empty ones in this repo. The directories are:
\Logs - Log files are written here
\Output - Output of experiments go here. The folder structure is as follows
	\Earthquake
		\DecisionTree
		\More Learners...
	\Diabetic Retinopathy
		\DecisionTree
		\More Learners
Experiment outputs that utilize all learners will be saved to the \Output\<DatasetName> root directory.
