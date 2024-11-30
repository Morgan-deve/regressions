import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
from dython.nominal import associations
# from utils import *

def load_data(filename, plot_=False):
    dataset = pd.read_csv(filename)

    name = filename.split('.')[0]
    if name == 'advertising':
        print(dataset.sample(5))
        print('=======================================')

        print("Number of samples:", dataset.shape[0])
        print('=======================================')


        mean_sales = dataset['sales'].mean()
        std_sales = dataset['sales'].std()
        print("Mean sales:", mean_sales)
        print("Standard deviation of sales:", std_sales)
        print('=======================================')

        correlation_matrix = dataset.corr()
        sales_correlation = correlation_matrix['sales']

        strongest_feature = sales_correlation.drop('sales').idxmax()
        strongest_correlation_value = sales_correlation[strongest_feature]

        print("Feature with the strongest linear relationship with sales:", strongest_feature)
        print("Correlation coefficient:", strongest_correlation_value)
        print('=======================================')
        dataset = dataset[['TV', 'sales']]

        train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

        # print("Number of samples in training set:", train_set.shape[0])
        # print("Number of samples in testing set:", test_set.shape[0])

        scaler = StandardScaler()
        train_set['TV'] = scaler.fit_transform(train_set[['TV']])
        test_set['TV'] = scaler.transform(test_set[['TV']])


        # print("\nNormalized Training Set:\n", train_set.head())
        # print("\nNormalized Testing Set:\n", test_set.head())

        # Prepare data for linear regression
        X_train = train_set['TV'].values.reshape(-1, 1)  # Feature matrix for training
        y_train = train_set['sales'].values  # Target variable for training

        X_test = test_set['TV'].values.reshape(-1, 1)  # Feature matrix for testing
        y_test = test_set['sales'].values  # Target variable for testing
        if plot_:
            # Create scatter plot for TV vs Sales
            plt.figure(figsize=(10, 5))
            plt.scatter(dataset['TV'], dataset['sales'], color='blue', alpha=0.5)
            plt.title(f'Scatter Plot of TV Advertising vs sales')
            plt.xlabel(f'TV Advertising Expenditure ($)')
            plt.ylabel('sales (in thousands)')
            plt.grid(True)
            plt.show()

            # Create scatter plot for radio vs Sales
            plt.figure(figsize=(10, 5))
            plt.scatter(dataset['radio'], dataset['sales'], color='orange', alpha=0.5)
            plt.title('Scatter Plot of radio Advertising vs sales')
            plt.xlabel('radio Advertising Expenditure ($)')
            plt.ylabel('sales (in thousands)')
            plt.grid(True)
            plt.show()

            # Create scatter plot for newspaper vs sales
            plt.figure(figsize=(10, 5))
            plt.scatter(dataset['newspaper'], dataset['sales'], color='green', alpha=0.5)
            plt.title('Scatter Plot of newspaper Advertising vs sales')
            plt.xlabel('newspaper Advertising Expenditure ($)')
            plt.ylabel('sales (in thousands)')
            plt.grid(True)
            plt.savefig("Newspaper.png")
            plt.show()
    
    elif name == 'weatherAUS':
        print(dataset.head())
        print(dataset.info())       #features type
        cat_features = dataset.select_dtypes(include=['object', 'category']).columns.tolist()
        cat_features.pop(0)
        
        print("Categorical Features:", cat_features)
        #report 7 tops correlations
        correlation_matrix = associations(dataset, nominal_columns='auto', numerical_columns=None)
        top_features = correlation_matrix['corr']['RainTomorrow'].abs().sort_values(ascending=False)[:8]
        print(top_features)
        
        #missing values
        missing_values = dataset.isnull().sum()
        missing_column = missing_values[missing_values > 0]
        print(f"features with missing valus: {missing_column}")
        
        #duplicated values
        duplicate = dataset.duplicated().sum()
        print(f"features with duplicate values: {duplicate}")
        
        #remove missing values
        initial_data = dataset.shape[0]
        data_cleaned = dataset.dropna()
        final_count = data_cleaned.shape[0]
        removed_count = initial_data - final_count
        print(f"the number of samples removed due to missing values: {removed_count}")
        
        #categoracalize the data
        categorical_features = data_cleaned.select_dtypes(include=['object']).columns
        label_encoder = LabelEncoder()
        for features in categorical_features:
            data_cleaned[features] = label_encoder.fit_transform(data_cleaned[features])
        print(data_cleaned.head())
        
        top_features_name = top_features.index[:-1]
        selected_featuers = list(top_features_name)+ ['RainTomorrow']
        final_dataset = data_cleaned[selected_featuers]
        print(f"shape of the final dataset: {final_dataset.shape[0]}")
        
        
    
    elif name == 'loan_risk_dataset1':
        print(dataset.head())
        cat_features = dataset.select_dtypes(include=['object', 'category']).columns.tolist()
        cat_features.pop(0)
        
        num_samples = dataset.shape[0]
        print(f"the number of samples: {num_samples}")
        
        mean_std = dataset.describe().loc[['mean', 'std']]
        print(f"the mean and standard deviation: {mean_std}")
        
        #missing and duplicates data
        missing_values = dataset.isnull().sum()
        missing_columns = missing_values[missing_values > 0]
        print(f"the missing values are: {missing_columns}")
        
        duplicates = dataset.duplicated().sum()
        print(f"the number of duplicates: {duplicates}")
          
        data_cleanned = dataset.dropna().drop_duplicates() #in 3rd dataset we have data_cleanned
        removed_counts = num_samples - data_cleanned.shape[0]
        print(f"Number of samples removed due to missing values or duplicates: {removed_counts}")
        
        #categorical features
        categorical_featuress = data_cleaned.select_dtypes(include=['object']).columns
        label_encoder = LabelEncoder()
        for features in categorical_featuress:
            data_cleanned[features] = label_encoder.fit_transform(data_cleaned[features])
        print(data_cleanned.head())
        
        #normalization
        normalized_data = (dataset - dataset.min() / (dataset.max() - dataset.min()))
        print(normalized_data.head())
        
        #z_score
        z_scores = np.abs((data_cleanned - data_cleanned.mean()) / data_cleanned.std())
        outliers = (z_scores > 2.6).any(axis=1)
        num_outliers = outliers.sum()
        print(f"Number of outliers detected with threshold 2.6: {num_outliers}")
        
        # Remove outliers from the dataset
        data_no_outliers = data_cleanned[~outliers]
        removed_outliers_count = data_cleanned.shape[0] - data_no_outliers.shape[0]
        print(f"Number of samples removed due to outlier detection: {removed_outliers_count}")
        
              
        
        
    return X_train, y_train, X_test, y_test, dataset







class Regressor:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        
        
# Linear Regression with Gradient Descent
class LinearRegressionGD(Regressor):
    def __init__(self,) :
        self.theta_0 = 0  
        self.theta_1 = 0  
        self.cost_history = []  
        super(LinearRegressionGD, self).__init__()
        
        
    def compute_cost(self, X, y):
        m = len(y)
        y_predicted = self.theta_0 + self.theta_1 * X.flatten()
        cost = (1/m) * np.sum((y - y_predicted) ** 2)
        return cost

    def fit(self, X, y):
        m = len(y)  
        for i in range(self.n_iterations):
            y_predicted = self.theta_0 + self.theta_1 * X.flatten()
            # Calculate gradients
            d_theta_0 = (-2/m) * np.sum(y - y_predicted)
            d_theta_1 = (-2/m) * np.sum((y - y_predicted) * X.flatten())
            
            self.theta_0 -= self.learning_rate * d_theta_0
            self.theta_1 -= self.learning_rate * d_theta_1
            
            
            if i % 100 == 0:
                cost = self.compute_cost(X, y)
                self.cost_history.append(cost)

    def predict(self, X):
        return self.theta_0 + self.theta_1 * X.flatten()
    
    
class LogesticRegressionGD(Regressor):
    def __init__(self,) :
        self.theta_0 = 0  
        self.theta_1 = 0  
        self.cost_history = []  
        super(LogesticRegressionGD, self).__init__()
        
    def _compute_cost(self, X, y):
        m = len(y)
        y_predicted = self.theta_0 + self.theta_1 * X.flatten()
        cost = (1/m) * np.sum((y - y_predicted) ** 2)
        return cost

    def fit(self, X, y):
        m = len(y)  
        for i in range(self.n_iterations):
            y_predicted = self.theta_0 + self.theta_1 * X.flatten()
            # Calculate gradients
            d_theta_0 = (-2/m) * np.sum(y - y_predicted)
            d_theta_1 = (-2/m) * np.sum((y - y_predicted) * X.flatten())
            
            self.theta_0 -= self.learning_rate * d_theta_0
            self.theta_1 -= self.learning_rate * d_theta_1
            
            
            if i % 100 == 0:
                cost = self._compute_cost(X, y)
                self.cost_history.append(cost)

    def predict(self, X):
        return self.theta_0 + self.theta_1 * X.flatten()

def main():
    
    X_train, y_train, X_test, y_test, dataset = load_data("weatherAUS.csv")
    # Initialize 
    model = LinearRegressionGD(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train, y_train)


    y_predicted_train = model.predict(X_train)
    y_predicted_test = model.predict(X_test)


    plt.figure(figsize=(10, 5))
    plt.plot(range(0, model.n_iterations, 100), model.cost_history, marker='o')
    plt.title('Cost Function Over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.show()

    # Plotting the regression line 
    plt.figure(figsize=(10, 5))
    plt.scatter(X_train, y_train, color='blue', label='Training Data', alpha=0.5)
    plt.scatter(X_test, y_test, color='orange', label='Testing Data', alpha=0.5)

    # Regression line for training data
    plt.plot(X_train, y_predicted_train, color='green', label='Regression Line (Training)', linewidth=2)

    # Regression line for testing data (optional)
    plt.plot(X_test, y_predicted_test, color='red', linestyle='--', label='Regression Line (Testing)', linewidth=2)

    plt.title('Linear Regression: Training and Testing Samples')
    plt.xlabel('Normalized TV Advertising Expenditure')
    plt.ylabel('Sales (in thousands)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Displaying coefficients for verification
    print("Intercept (theta_0):", model.theta_0)
    print("Slope (theta_1):", model.theta_1)

    # Display first few predictions alongside actual values for comparison
    predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_predicted_test})
    print(predictions_df.head())


if __name__ == '__main__':
    main()