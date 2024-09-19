
from typing import Union
import pandas as pd


def extract_data(dataset: pd.DataFrame, columns: Union[list[str], str]):
    return dataset[columns].values

def load_data(path: str) -> pd.DataFrame:
        try:
            data = pd.read_csv(path)

        except FileNotFoundError:
            if(path.__contains__('../')):
                path = path.replace('../', '')
                data = pd.read_csv(path)
            else:
                path = '../' + path
                pd.read_csv(path)

        return data