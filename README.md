Dataset Source- https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset/data
This data is from the 2011 Kaggle Competition: Give Me Some Credit.

File Description-
    For Jupyter-
        JupyterNotebook.ipynb

        
    For Python/ VSCode app-
        1. ImportLibraries.py     -  Containes all the libraries/modules used in this project.
        2. DataCleaning.py        -  Reads the training datafile(csv), imputes missing values, removes outliers.
                                     Removes not required features, Normalizes the dataframe and returns the scaled 
                                     dataframe.
        3. Logistic Regression.py -  Splits the features into predictor_variable and response variable.
                                     Randomly splits the entire dataset into training & testing set in the ratio of 70-30.
                                     Applies Logistic Regression model and creates a response prediction.
                                     Creates a confusion matrix using response prediction & Response Test variable
                                     Shows the accuracy of the model
        4. TestDataset.py         -  Reads training dataset, normalizes the features and splits into prediction and 
                                     response variable.
                                     Reads testing dataset,selects only necessary features which are same as training
                                     dataset, imputes missing value.
                                     Fits the predictor features and response feature to form a model.
                                     Prints out accuracy of the model.
