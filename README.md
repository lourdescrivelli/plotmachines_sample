# plotmachines_sample
Testing Reproducibility of PlotMachines Paper (https://arxiv.org/abs/2004.14967)

Custom preprocessing for the PlotMachines Paper . Original code can be found at: https://github.com/hrashkin/plotmachines  

### What is different?  

#### Sample Database
The original paper uses Wikiextractor. Currently, there seems to be a bug with this repository, so I was unable to reproduce the original data.  
Instead, I have used a sample plots and titles database provided by Wikiplots (https://github.com/markriedl/WikiPlots)
Since there is no concept of paragraphs in the sample database, I added a custom function to arbitrary split the paragraphs into Introduction, Body and Conclusion. 

[pm_preprocessing.py] builds on the original code from [extract_outlines.py] so that it can work with the new database.  
The original [extract_outlines.py] can be found at : https://github.com/hrashkin/plotmachines/blob/master/src/preprocessing/extract_outlines.py  

#### Pickle function
Adds the last step for preprocessing, currently missing form the original code. To run this code you will need the [model.py] function from the original repository. https://github.com/hrashkin/plotmachines/blob/master/src/model/model.py
