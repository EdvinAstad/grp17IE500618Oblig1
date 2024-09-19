from enum import Enum
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

class ModelType(Enum):
    LINEAR_REGRESSION = LinearRegression
    RANDOM_FOREST_REGRESSION = RandomForestRegressor
