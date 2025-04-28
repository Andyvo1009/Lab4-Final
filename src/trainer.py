from pyspark.context import SparkContext
from pyspark.streaming.context import StreamingContext
from pyspark.sql.context import SQLContext
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import IntegerType, StructField, StructType
from pyspark.ml.linalg import VectorUDT
from transforms import Transforms
import pyspark

import numpy as np
import os
transforms_file = os.path.abspath("src/transforms.py")
class SparkConfig:
    appName = "CancerDataSVM"
    receivers = 4  # From main.py
    host = "local"
    stream_host = "localhost"
    port = 6100
    batch_interval = 10  # From main.py

from dataloader import DataLoader

class Trainer:
    def __init__(self, 
                 model, 
                 split: str, 
                 spark_config: SparkConfig, 
                 transforms: Transforms):
        self.model = model
        self.split = split
        self.sparkConf = spark_config
        self.transforms = transforms
        self.sc = SparkContext(f"{self.sparkConf.host}[{self.sparkConf.receivers}]", f"{self.sparkConf.appName}")
        self.ssc = StreamingContext(self.sc, self.sparkConf.batch_interval)
        self.sqlContext = SQLContext(self.sc)
        self.dataloader = DataLoader(self.sc, self.ssc, self.sqlContext, self.sparkConf, self.transforms)
        self.total_batches = 0
        self.cm = 0.0
        self.test_accuracy = 0.0
        self.test_loss = 0.0
        self.test_precision = 0.0
        self.test_recall = 0.0
        self.test_f1 = 0.0
        self.batch_count = 0
        self.sc.addPyFile(transforms_file)
    def train(self):
        stream = self.dataloader.parse_stream()
        stream.foreachRDD(self.__train__)
        self.ssc.start()
        self.ssc.awaitTermination()

    def __train__(self, timestamp, rdd: pyspark.RDD) -> DataFrame:
        print(f"RDD: {rdd.take(1)}")
        if not rdd.isEmpty():
            sample = rdd.take(1)[0]
            print(f"Features type: {type(sample[0])}, Features shape: {np.array(sample[0]).shape if isinstance(sample[0], (list, np.ndarray)) else 'N/A'}")

            schema = StructType([
                StructField("features", VectorUDT(), True),  # 30 numerical features
                StructField("label", IntegerType(), True)
            ])

            try:
                df = self.sqlContext.createDataFrame(rdd, schema)
                print(f"DataFrame created: {df.take(1)}")
                predictions, accuracy, precision, recall, f1 = self.model.train(df)
                print("="*10)
                print(f"Predictions = {predictions}")
                print(f"Accuracy = {accuracy:.2f}")
                print(f"Precision = {precision:.2f}")
                print(f"Recall = {recall:.2f}")
                print(f"F1 Score = {f1:.2f}")
                print("="*10)
            except Exception as e:
                print(f"Error creating DataFrame or training: {e}")
        self.total_batches += rdd.count()
        print("Total Batch Size of RDD Received:", rdd.count())
        print("+"*20)

    def predict(self):
        stream = self.dataloader.parse_stream()
        stream.foreachRDD(self.__predict__)
        self.ssc.start()
        self.ssc.awaitTermination()

    def __predict__(self, rdd: pyspark.RDD) -> DataFrame:
        self.batch_count += 1
        if not rdd.isEmpty():
            schema = StructType([
                StructField("features", VectorUDT(), True),
                StructField("label", IntegerType(), True)
            ])
            try:
                df = self.sqlContext.createDataFrame(rdd, schema)
                accuracy, loss, precision, recall, f1, cm = self.model.predict(df, self.model)
                self.cm += cm
                self.test_accuracy += accuracy / max(self.total_batches, 1)
                self.test_loss += loss / max(self.total_batches, 1)
                self.test_precision += precision / max(self.total_batches, 1)
                self.test_recall += recall / max(self.total_batches, 1)
                self.test_f1 += f1 / max(self.total_batches, 1)
                print(f"Test Accuracy: {self.test_accuracy:.2f}")
                print(f"Test Loss: {self.test_loss:.2f}")
                print(f"Test Precision: {self.test_precision:.2f}")
                print(f"Test Recall: {self.test_recall:.2f}")
                print(f"Test F1 Score: {self.test_f1:.2f}")
                print(f"Confusion matrix: {cm}")
            except Exception as e:
                print(f"Error in prediction: {e}")
        print(f"batch: {self.batch_count}")
        print("Total Batch Size of RDD Received:", rdd.count())
        print("---------------------------------------")