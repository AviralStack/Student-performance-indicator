import sys
import os
from dataclasses import dataclass

import pandas as pd

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformation:
    dataset: str
    output_feature: str

    def transformation(self):
        logging.info("Entering Data Transformation stage")

        try:
            file_type = os.path.splitext(self.dataset)[1]

            # Load dataset
            if file_type == '.csv':
                df = pd.read_csv(self.dataset)
            elif file_type in ['.xls', '.xlsx']:
                df = pd.read_excel(self.dataset)
            else:
                raise Exception("Unsupported file format")

            logging.info("Dataset loaded successfully")

            # Split X and y
            X = df.drop(self.output_feature, axis=1)
            y = df[self.output_feature]

            num_features = X.select_dtypes(exclude="object").columns
            cat_features = X.select_dtypes(include="object").columns

            from sklearn.preprocessing import OneHotEncoder, StandardScaler
            from sklearn.compose import ColumnTransformer

            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder(handle_unknown='ignore')

            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", oh_transformer, cat_features),
                    ("StandardScaler", numeric_transformer, num_features),
                ]
            )

      
            X_transformed = preprocessor.fit_transform(X)

            oh_features = preprocessor.named_transformers_["OneHotEncoder"] \
                .get_feature_names_out(cat_features)

            all_features = list(oh_features) + list(num_features)

            # Convert to DataFrame
            X = pd.DataFrame(X_transformed, columns=all_features)

            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            logging.info("Data transformation completed successfully")

            return X_train, X_test, y_train, y_test, file_type

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        logging.info("Starting data transformation pipeline")

        dataset_path = "/home/aviral-linux/Student-Performance/artifacts/train.csv"   
        target_column = "math_score"           

        transformer = DataTransformation(
            dataset=dataset_path,
            output_feature=target_column
        )

        X_train, X_test, y_train, y_test, file_type = transformer.transformation()

        
        output_dir = "/home/aviral-linux/Student-Performance/split_dataset"
        os.makedirs(output_dir, exist_ok=True)

    
        y_train = pd.DataFrame(y_train, columns=[target_column])
        y_test = pd.DataFrame(y_test, columns=[target_column])

        if file_type == ".csv":
            X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
            X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
            y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
            y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

        elif file_type in [".xls", ".xlsx"]:
            X_train.to_excel(os.path.join(output_dir, "X_train.xlsx"), index=False)
            X_test.to_excel(os.path.join(output_dir, "X_test.xlsx"), index=False)
            y_train.to_excel(os.path.join(output_dir, "y_train.xlsx"), index=False)
            y_test.to_excel(os.path.join(output_dir, "y_test.xlsx"), index=False)

        else:
            raise Exception("Unsupported file format")

        logging.info("All datasets saved successfully in split_dataset folder")

    except Exception as e:
        raise CustomException(e, sys)