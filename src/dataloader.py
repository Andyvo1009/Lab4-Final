import json
import numpy as np
from pyspark.context import SparkContext
from pyspark.sql.context import SQLContext
from pyspark.streaming.context import StreamingContext
from pyspark.streaming.dstream import DStream
from pyspark.ml.linalg import DenseVector
from transforms import Transforms
from trainer import SparkConfig

class DataLoader:
    def __init__(self, 
                 sparkContext: SparkContext, 
                 sparkStreamingContext: StreamingContext, 
                 sqlContext: SQLContext,
                 sparkConf: SparkConfig, 
                 transforms: Transforms):
        self.sc = sparkContext
        self.ssc = sparkStreamingContext
        self.sparkConf = sparkConf
        self.sql_context = sqlContext
        self.stream = self.ssc.socketTextStream(
            hostname=self.sparkConf.stream_host, 
            port=self.sparkConf.port
        )
        self.transforms = transforms

    def parse_stream(self) -> DStream:
        def safe_json_load(line):
            try:
                return json.loads(line)
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}, line: {line}")
                return {}

        json_stream = self.stream.map(safe_json_load)
        json_stream_exploded = json_stream.flatMap(lambda x: x.values() if x else [])
        json_stream_exploded.foreachRDD(lambda rdd: print(f"Raw JSON data: {len(rdd.collect())} items"))

        def extract_features(x):
            try:
                features = [x[f'feature-{i}'] for i in range(30)]  # 30 numerical features
                label = x['label']
                return [features, label]
            except Exception as e:
                print(f"Extraction error: {e}, data: {x}")
                return [[0.0] * 30, 0]

        features = json_stream_exploded.map(extract_features)
        features.foreachRDD(lambda rdd: print(f"Extracted features: {rdd.take(1)}"))
        features = DataLoader.preprocess(features, self.transforms)
        return features

    @staticmethod
    def preprocess(stream: DStream, transforms: Transforms) -> DStream:
        stream = stream.map(lambda x: [transforms.transform(x[0]), x[1]])
        stream = stream.map(lambda x: [DenseVector(x[0]), x[1]])
        return stream