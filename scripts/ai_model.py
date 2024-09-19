import array
from enum import Enum
from matplotlib import pyplot as plt
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from model_type import ModelType
from data_loader import load_data, extract_data
from typing_extensions import Self
from key_words import Keywords


class AiModel(): 
    """
    A class representing a machine learning model with functionalities to train, evaluate, and display results.
    
    Attributes:
    ----------
    _model_type : ModelType
        The type of model being used (e.g., Linear Regression, Random Forest Regression).
    _model : RegressorMixin
        The machine learning model instance.
    _training_features : array
        Features used for training the model.
    _training_target : array
        Target values used for training the model.
    _testing_features : array
        Features used for testing the model.
    _testing_target : array
        Target values used for testing the model.
    _test_data_target_predictions : array
        Predictions made by the model on the test data.
    _meta_data : dict
        Metadata related to the model and its training.
    _r_value : float
        R-squared value of the model on test data.
    _mean_square_error : float
        Mean squared error of the model on test data.
    
    Methods:
    -------
    __init__(model_type: ModelType, target: array, features: array, meta_data: dict = dict, **kwargs):
        Initializes the AiModel with the specified model type, target, features, and metadata.
    print_meta_data():
        Prints the metadata of the model.
    _get_specifications(self) -> dict
        Filters meta_data for arguments used in constructor for model.
    calculate_r_squared() -> float:
        Calculates and returns the R-squared value of the model.
    calculate_mean_square_error() -> float:
        Calculates and returns the mean squared error of the model.
    display(x_axis_label: str = 'Features', y_axis_label: str = 'SalePrice'):
        Displays a plot comparing actual vs. predicted values, along with model performance metrics.
    
    Properties:
    -----------
    model: RegressorMixin
        The model used for prediction. Can be set to an instance of RegressorMixin.
    model_type: ModelType
        The type of model being used.
    training_features: array
        The features used for training the model.
    training_target: array
        The target values used for training the model.
    testing_features: array
        The features used for testing the model.
    testing_target: array
        The target values used for testing the model.
    test_data_target_predictions: array
        Predictions made by the model on the test data.
    meta_data: dict
        Metadata related to the model and its training.
    r_value: float
        R-squared value of the model on test data.
    mean_square_error: float
        Mean squared error of the model on test data.
    
    Static Methods:
    ---------------
    splitt_training_testing_data(features: array, target: array) -> tuple:
        Splits the features and target into training and testing sets.

    assemble() -> 'AiModel.Builder'
        Return builder object.
    """
        
    _model_type: ModelType
    _model: RegressorMixin
    _training_features: array
    _training_target: array
    _testing_features: array
    _testing_target: array
    _test_data_target_predictions: array
    _meta_data: dict
    _r_value: float
    _mean_square_error: float

    def __init__(self, model_type: ModelType, target: array, features: array, meta_data: dict = dict, **kwargs):
        """
        Initializes the AiModel with the specified model type, target, features, and metadata.

        Parameters:
        ----------
        model_type : ModelType
            The type of model to be used (e.g., Linear Regression, Random Forest Regression).
        target : array
            Target values for the model.
        features : array
            Features for the model.
        meta_data : dict, optional
            Additional metadata related to the model (default is an empty dictionary).
        **kwargs:
            Additional keyword arguments to be added to meta_data.
        """
        self.training_features, self.testing_features, self.training_target, self.testing_target = self.splitt_training_testing_data(features, target)
        meta_data.update(kwargs)
        self.meta_data = meta_data
        self.model_type = model_type
        self.model = self.model_type.value(**self._get_specifications())
        self.model.fit(self.training_features, self.training_target)
        self.test_data_target_predictions = self.model.predict(self.testing_features)
        self.mean_square_error = self.calculate_mean_square_error()
        self.r_value = self.calculate_r_squared()

    def _get_specifications(self) -> dict:
        """
        Filters metadata and returns a dictionary of valid keyword arguments based on the model type.

        This method checks the model type (`LINEAR_REGRESSION` or `RANDOM_FOREST_REGRESSION`) and retrieves
        the appropriate set of keywords for that model type. It then iterates over the `meta_data` dictionary
        and selects only the entries whose keys match the valid keywords for the current model type. The filtered
        metadata is returned as a dictionary.

        Returns:
            kwargs (dict): A dictionary containing the filtered keyword arguments based on the model type and 
            the keys in `meta_data`.

        Raises:
            None
        """
        kwargs = {}
        if(self.model_type == ModelType.LINEAR_REGRESSION):
            keywords = Keywords.LINEAR_REAGRESSION_KEYWORDS.value

        elif(self.model_type == ModelType.RANDOM_FOREST_REGRESSION):
            keywords = Keywords.RANDOM_FOREST_KEYWORDS.value

        for key, value in self.meta_data.items():
                if key in keywords:
                    kwargs[key] = value

        return kwargs
    
    def print_meta_data(self):
        """
        Prints the metadata of the model.
        """
        print(self._meta_data)

    @property
    def model(self) -> RegressorMixin:
        """
        Gets the model used for prediction.

        Returns:
        -------
        RegressorMixin
            The model instance.
        """
        return self._model

    @model.setter
    def model(self, value: RegressorMixin):
        """
        Sets the model for prediction.

        Parameters:
        ----------
        value : RegressorMixin
            The model instance to be set.
        """
        self._model = value

    @property
    def model_type(self) -> ModelType:
        """
        Gets the type of model being used.

        Returns:
        -------
        ModelType
            The type of model.
        """
        return self._model_type

    @model_type.setter
    def model_type(self, value: ModelType):
        """
        Sets the type of model being used.

        Parameters:
        ----------
        value : ModelType
            The type of model to be set.
        """
        self._model_type = value

    @property
    def training_features(self) -> array:
        """
        Gets the features used for training.

        Returns:
        -------
        array
            The training features.
        """
        return self._training_features

    @training_features.setter
    def training_features(self, value: array):
        """
        Sets the features used for training.

        Parameters:
        ----------
        value : array
            The training features to be set.
        """
        self._training_features = value

    @property
    def training_target(self) -> array:
        """
        Gets the target values used for training.

        Returns:
        -------
        array
            The training target values.
        """
        return self._training_target

    @training_target.setter
    def training_target(self, value: array):
        """
        Sets the target values used for training.

        Parameters:
        ----------
        value : array
            The training target values to be set.
        """
        self._training_target = value

    @property
    def testing_features(self) -> array:
        """
        Gets the features used for testing.

        Returns:
        -------
        array
            The testing features.
        """
        return self._testing_features

    @testing_features.setter
    def testing_features(self, value: array):
        """
        Sets the features used for testing.

        Parameters:
        ----------
        value : array
            The testing features to be set.
        """
        self._testing_features = value

    @property
    def testing_target(self) -> array:
        """
        Gets the target values used for testing.

        Returns:
        -------
        array
            The testing target values.
        """
        return self._testing_target

    @testing_target.setter
    def testing_target(self, value: array):
        """
        Sets the target values used for testing.

        Parameters:
        ----------
        value : array
            The testing target values to be set.
        """
        self._testing_target = value

    @property
    def test_data_target_predictions(self) -> array:
        """
        Gets the predictions made by the model on the test data.

        Returns:
        -------
        array
            The test data target predictions.
        """
        return self._test_data_target_predictions

    @test_data_target_predictions.setter
    def test_data_target_predictions(self, value: array):
        """
        Sets the predictions made by the model on the test data.

        Parameters:
        ----------
        value : array
            The test data target predictions to be set.
        """
        self._test_data_target_predictions = value

    @property
    def meta_data(self) -> dict:
        """
        Gets the metadata related to the model.

        Returns:
        -------
        dict
            The metadata dictionary.
        """
        return self._meta_data

    @meta_data.setter
    def meta_data(self, value: dict):
        """
        Sets the metadata related to the model.

        Parameters:
        ----------
        value : dict
            The metadata dictionary to be set.
        """
        self._meta_data = value

    @property
    def r_value(self) -> float:
        """
        Gets the R-squared value of the model.

        Returns:
        -------
        float
            The R-squared value.
        """
        return self._r_value

    @r_value.setter
    def r_value(self, value: float):
        """
        Sets the R-squared value of the model.

        Parameters:
        ----------
        value : float
            The R-squared value to be set.
        """
        self._r_value = value

    @property
    def mean_square_error(self) -> float:
        """
        Gets the mean squared error of the model.

        Returns:
        -------
        float
            The mean squared error.
        """
        return self._mean_square_error

    @mean_square_error.setter
    def mean_square_error(self, value: float):
        """
        Sets the mean squared error of the model.

        Parameters:
        ----------
        value : float
            The mean squared error to be set.
        """
        self._mean_square_error = value
       
    @staticmethod
    def splitt_training_testing_data(features: array, target: array) -> tuple:
        """
        Splits the provided features and target arrays into training and testing sets.
        
        Parameters:
        ----------
        features : array
            The features to be split into training and testing sets.
        target : array
            The target values to be split into training and testing sets.
        
        Returns:
        -------
        tuple
            A tuple containing four elements: 
            - The training features (array)
            - The testing features (array)
            - The training target values (array)
            - The testing target values (array)
        """
        features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)
        return features_train, features_test, target_train, target_test

    def calculate_r_squared(self) -> float:
        """
        Calculates the R-squared value of the model based on the testing data and the model's predictions.
        
        Returns:
        -------
        float
            The R-squared value of the model, representing the proportion of variance in the testing target that is predictable from the testing features.
        """
        return mean_squared_error(self.testing_target, self.test_data_target_predictions)

    def calculate_mean_square_error(self) -> float:
        """
        Calculates the mean squared error of the model based on the testing features and target values.
        
        Returns:
        -------
        float
            The mean squared error of the model, representing the average squared difference between the actual and predicted target values.
        """
        return self.model.score(self.testing_features, self.testing_target)

    def display(self, x_axis_label: str = 'Features', y_axis_label: str = 'SalePrice'):
        """
        Displays a plot comparing the actual vs. predicted values from the model along with the model type, mean squared error, and R-squared value.
        
        Parameters:
        ----------
        x_axis_label : str, optional
            The label for the x-axis of the plot (default is 'Features').
        y_axis_label : str, optional
            The label for the y-axis of the plot (default is 'SalePrice').
        
        Returns:
        -------
        None
            This method does not return any value; it only displays a plot.
        """
        print(f"model type: {self.model_type}")
        print(f"Mean Squared Error: {self.mean_square_error}")
        print(f"R-squared: {self.r_value}")

        y_target = self.testing_target
        y_pred = self.test_data_target_predictions

        sorted_indices = np.argsort(y_target)
        sorted_y_test = y_target[sorted_indices]
        sorted_y_pred = y_pred[sorted_indices]

        plt.figure(figsize=(12, 6))
        plt.plot(sorted_y_test, label='Actual Values', color='blue', marker='o', linestyle='-', markersize=5)
        plt.plot(sorted_y_pred, label='Predicted Values', color='red', marker='x', linestyle='', markersize=5)
        plt.xlabel(x_axis_label)
        plt.ylabel(y_axis_label)
        plt.title('Comparison of Actual vs. Predicted Values')
        plt.legend()
        plt.show()

    @staticmethod
    def assemble() -> 'AiModel.Builder':
        """
        Description:
        The assemble method is a static method that serves as a 
        convenient entry point to instantiate the nested AiModel.Builder class. 
        It returns an instance of AiModel.Builder, enabling the construction of an AiModel object through the builder pattern. 
        The method calls the private class method _create_instance to initialize the builder.

        Returns:
            AiModel.Builder: An instance of the nested AiModel.Builder class.
        Usage:
            This method allows you to easily start the builder pattern process for creating an AiModel object. 
            It abstracts the details of how the builder is instantiated.
        """
        return AiModel.Builder._create_instance()
            


    class Builder:
        """
        A builder class for constructing an `AiModel` instance with various configuration options.
        
        Attributes:
        ----------
        _model_type : ModelType
            The type of model to be used (e.g., LinearRegression, RandomForestRegressor).
        _features : array
            The features to be used for training and testing.
        _target : array
            The target values for training and testing.
        _meta_data : dict
            Metadata associated with the model.
        
        Methods:
        -------
        Private Properties:

            _create_instance() -> 'AiModel.Builder'
                Creates and returns a new instance of the Builder class.
            
            __model_type__() -> ModelType
                Gets the model type.
            
            __model_type__(value: ModelType)
                Sets the model type.
            
            __features__() -> array
                Gets the features.
            
            __features__(value: array)
                Sets the features.
            
            __target__() -> array
                Gets the target values.
            
            __target__(value: array)
                Sets the target values.
            
            __meta_data__() -> dict
                Gets the metadata.
            
            __meta_data__(value: dict)
                Sets the metadata.
            
            __init__()
                Initializes the Builder with default settings.
        
        load_features(features_list: list[str], path: str = '../data/raw/AmesHousing.csv') -> Self
            Loads features from a CSV file and updates the `features` attribute and `meta_data`.
        
        load_target(target_str: str = 'SalePrice', path: str = '../data/raw/AmesHousing.csv') -> Self
            Loads target values from a CSV file and updates the `target` attribute and `meta_data`.
        
        set_model_type(value: ModelType) -> Self
            Sets the `model_type` attribute and returns the builder instance.
        
        add_training_features_array(value: array) -> Self
            Adds an array of training features and returns the builder instance.
        
        add_training_target_array(value: array) -> Self
            Adds an array of training target values and returns the builder instance.
        
        add_meta_data(key: str, value: str) -> Self
            Adds a key-value pair to the `meta_data` dictionary and returns the builder instance.
        
        build() -> 'AiModel'
            Constructs and returns an `AiModel` instance with the current configuration.
        """

        _model_type: ModelType
        _features: array
        _target: array
        _meta_data: dict

        def __init__(self):
            """
            Initializes the Builder with default settings.
            """
            self.__meta_data__ = dict()
            self.__model_type__ = ModelType.LINEAR_REGRESSION
            self.load_target()
            self.load_features(['Overall Qual'])

        @classmethod
        def _create_instance(cls) -> 'AiModel.Builder':
            """
            Creates and returns a new instance of the Builder class.

            Returns:
            -------
            AiModel.Builder
                A new instance of the Builder class.
            """
            return cls()
        
        @property
        def __model_type__(self) -> ModelType:
            """
            Gets the model type.

            Returns:
            -------
            ModelType
                The model type.
            """
            return self._model_type

        @__model_type__.setter
        def __model_type__(self, value: ModelType):
            """
            Sets the model type.

            Parameters:
            ----------
            value : ModelType
                The model type to be set.
            """
            self._model_type = value

        @property
        def __features__(self) -> array:
            """
            Gets the features.

            Returns:
            -------
            array
                The features.
            """
            return self._features

        @__features__.setter
        def __features__(self, value: array):
            """
            Sets the features.

            Parameters:
            ----------
            value : array
                The features to be set.
            """
            self._features = value

        @property
        def __target__(self) -> array:
            """
            Gets the target values.

            Returns:
            -------
            array
                The target values.
            """
            return self._target

        @__target__.setter
        def __target__(self, value: array):
            """
            Sets the target values.

            Parameters:
            ----------
            value : array
                The target values to be set.
            """
            self._target = value

        @property
        def __meta_data__(self) -> dict:
            """
            Gets the metadata.

            Returns:
            -------
            dict
                The metadata.
            """
            return self._meta_data

        @__meta_data__.setter
        def __meta_data__(self, value: dict):
            """
            Sets the metadata.

            Parameters:
            ----------
            value : dict
                The metadata to be set.
            """
            self._meta_data = value
                    
        def load_features(self, features_list: list[str], path: str = '../data/raw/AmesHousing.csv') -> Self:
            """
            Loads features from a CSV file and updates the `features` attribute and `meta_data`.

            Parameters:
            ----------
            features_list : list[str]
                A list of feature names to be loaded.
            path : str, optional
                The path to the CSV file (default is '../data/raw/AmesHousing.csv').

            Returns:
            -------
            Self
                The current Builder instance.
            """
            self.__meta_data__['features'] = ', '.join(features_list)
            data = load_data(path)
            features_arr = extract_data(data, features_list)
            self.__features__ = features_arr
            return self

        def load_target(self, target_str: str = 'SalePrice', path: str = '../data/raw/AmesHousing.csv') -> Self:
            """
            Loads target values from a CSV file and updates the `target` attribute and `meta_data`.

            Parameters:
            ----------
            target_str : str, optional
                The name of the target column in the CSV file (default is 'SalePrice').
            path : str, optional
                The path to the CSV file (default is '../data/raw/AmesHousing.csv').

            Returns:
            -------
            Self
                The current Builder instance.
            """
            temp_meta_data = self.__meta_data__
            temp_meta_data['target'] = target_str 
            self.__meta_data__ = temp_meta_data 
            data = load_data(path)
            target_arr = extract_data(data, target_str)
            self.__target__ = target_arr
            return self
            
        def set_model_type(self, value: ModelType) -> Self:
            """
            Sets the `model_type` attribute and returns the builder instance.

            Parameters:
            ----------
            value : ModelType
                The model type to be set.

            Returns:
            -------
            Self
                The current Builder instance.
            """
            self.__model_type__ = value
            return self

        def add_training_features_array(self, value: array) -> Self:
            """
            Adds an array of training features and returns the builder instance.

            Parameters:
            ----------
            value : array
                The array of training features to be added.

            Returns:
            -------
            Self
                The current Builder instance.
            """
            self.__features__ = value
            return self
            
        def add_training_target_array(self, value: array) -> Self:
            """
            Adds an array of training target values and returns the builder instance.

            Parameters:
            ----------
            value : array
                The array of training target values to be added.

            Returns:
            -------
            Self
                The current Builder instance.
            """
            self.__target__ = value
            return self

        def add_meta_data(self, key: str, value: str) -> Self:
            """
            Adds a key-value pair to the `meta_data` dictionary and returns the builder instance.

            Parameters:
            ----------
            key : str
                The key for the metadata entry.
            value : str
                The value for the metadata entry.

            Returns:
            -------
            Self
                The current Builder instance.
            """
            temp_meta_data = self.__meta_data__
            temp_meta_data[key] = value
            self.__meta_data__ = temp_meta_data
            return self
            
        def build(self) -> 'AiModel':
            """
            Constructs and returns an `AiModel` instance with the current configuration.

            Returns:
            -------
            AiModel
                The constructed AiModel instance.
            """
            return AiModel(self.__model_type__, self.__target__, self.__features__, self.__meta_data__)