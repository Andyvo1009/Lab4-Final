import time
import json
import socket
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Streams Cancer Data to Spark Streaming Context')
parser.add_argument('--batch-size', '-b', help='Batch size', required=True, type=int)
parser.add_argument('--endless', '-e', help='Enable endless stream', type=bool, default=False)
parser.add_argument('--split', '-s', help="training or test split", type=str, default='train')
parser.add_argument('--sleep', '-t', help="streaming interval", type=int, default=3)

TCP_IP = "localhost"
TCP_PORT = 6100

class Dataset:
    def __init__(self):
        self.data = []
        self.labels = []
        df = pd.read_csv("datasets/Cancer_Data.csv")
        self.data = df.drop(columns=['id', 'diagnosis']).values
        self.labels = df['diagnosis'].map({'M': 1, 'B': 0}).values.tolist()

    def data_generator(self, batch_size):
        batch = []
        size_per_batch = (len(self.data) // batch_size) * batch_size
        for ix in range(0, size_per_batch, batch_size):
            features = self.data[ix:ix+batch_size]
            labels = self.labels[ix:ix+batch_size]
            batch.append([features, labels])
        return batch

    def sendCancerDataToSpark(self, tcp_connection, batch_size, split="train"):
        total_samples = 569  # Cancer Data has 569 samples
        total_batch = total_samples // batch_size + (1 if total_samples % batch_size else 0)
        pbar = tqdm(total=total_batch)
        data_received = 0

        batches = self.data_generator(batch_size)
        for batch in batches:
            features, labels = batch
            features = np.array(features)
            batch_size, feature_size = features.shape
            features = features.tolist()

            payload = dict()
            for batch_idx in range(batch_size):
                payload[batch_idx] = dict()
                for feature_idx in range(feature_size):
                    payload[batch_idx][f'feature-{feature_idx}'] = features[batch_idx][feature_idx]
                payload[batch_idx]['label'] = labels[batch_idx]

            payload = (json.dumps(payload) + "\n").encode()
            try:
                tcp_connection.send(payload)
            except (BrokenPipeError, ConnectionResetError) as e:
                print(f"Connection error: {e}. Reconnecting...")
                tcp_connection.close()
                tcp_connection, _ = self.connectTCP()
                tcp_connection.send(payload)
            except Exception as error_message:
                print(f"Exception thrown but was handled: {error_message}")

            data_received += 1
            pbar.update(n=1)
            pbar.set_description(f"it: {data_received} | received: {batch_size} samples")
            time.sleep(sleep_time)
        return tcp_connection

    def connectTCP(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((TCP_IP, TCP_PORT))
        s.listen(1)
        print(f"Waiting for connection on port {TCP_PORT}...")
        connection, address = s.accept()
        print(f"Connected to {address}")
        return connection, address

    def streamCancerDataset(self, tcp_connection, batch_size, train_test_split):
        tcp_connection = self.sendCancerDataToSpark(tcp_connection, batch_size, train_test_split)
        return tcp_connection

if __name__ == '__main__':
    args = parser.parse_args()
    batch_size = args.batch_size
    endless = args.endless
    sleep_time = args.sleep
    train_test_split = args.split
    dataset = Dataset()
    tcp_connection, _ = dataset.connectTCP()

    if endless:
        while True:
            tcp_connection = dataset.streamCancerDataset(tcp_connection, batch_size, train_test_split)
    else:
        tcp_connection = dataset.streamCancerDataset(tcp_connection, batch_size, train_test_split)
    print('Stop here')
    tcp_connection.close()