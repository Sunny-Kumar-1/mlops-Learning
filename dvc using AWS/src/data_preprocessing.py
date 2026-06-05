import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")

#ensureing lod dir exist
log_dir = "./dvc using AWS/logs"
os.makedirs(log_dir, exist_ok=True)

#setting up logger
logger = logging.getLogger("data_preprocessing")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, "data_processing.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """
    Transforms the input text by converting it to lowercase, tokenizing, removing stopwords and punctuation, and stemming.
    """
    ps = PorterStemmer()
    #convet to lowercase
    text=text.lower()
    #tokenize the text
    text = nltk.word_tokenize(text)
    #Remove stopwards and punctuation
    text=[word for word in text if word.isalnum()]
    #stem the words
    text = [ps.stem(word) for word in text]
    #join th token back into a single string
    return " ".join(text)

def preprocess_df(df, text_column="text", target_column="target"):
    """
    Preprocesses the DataFrame by encoding the target column, removing duplicates, and transforming the text column.
    """
    try:
        logger.debug("starting preprocessing for dataframe")
        #encode the target column
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug("target column encoded")

        #remove duplicate rows
        df = df.drop_duplicates(keep="first")
        logger.debug("duplicate removed")

        #apply text transformation to the specified text column
        df.loc[:,text_column] = df[text_column].apply(transform_text)
        logger.debug("text column transformed")
        return df
    
    except KeyError as e:
        logger.error("column not found: %s",e)
        raise
    except Exception as e:
        logger.error("error during text normalization: %s",e)
        raise

def main(text_column="text", target_column="target"):
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """
    try:
        #fetch the data from data/raw
        train_data = pd.read_csv("./dvc using AWS/data/raw/train.csv")
        test_data = pd.read_csv("./dvc using AWS/data/raw/test.csv")
        logger.debug("data loaded properly")

        #transform the data 
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        #store the data inside data/processed
        data_path = os.path.join("./dvc using AWS/data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path,"test_processed.csv"), index=False)

        logger.debug("processed data save to %s",data_path)
    except FileNotFoundError as e:
        logger.debug("file not found: %s",e)
    except pd.errors.EmptyDataError as e:
        logger.debug("no DATA : %s",e)
    except Exception as e:
        logger.error("failed to complete the data transformation process: %s",e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
