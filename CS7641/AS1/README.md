For Assignment 1 analysis I utilized sklearn. A lot of the code I used can be found in the skearn documentation. I've highlighted the URLs used in this assignment here:

The original datasets can be found at these links:

The modified datasets(Described in report) can be found in the Data folder.

To assist in running the experiment code, I've created a yml file. Simply create a virtual envirionment from the file, activate it, and you should be able to run the experiments.

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
