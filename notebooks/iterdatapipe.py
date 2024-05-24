import torch
from torch.utils.data import IterDataPipe

# Define a simple source DataPipe
class RangeDataPipe(IterDataPipe):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __iter__(self):
        return iter(range(self.start, self.end))

# Define a simple transform DataPipe
class SquareDataPipe(IterDataPipe):
    def __init__(self, source_datapipe):
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for x in self.source_datapipe:
            yield x * x

# Create a data pipeline
source_dp = RangeDataPipe(0, 10)
square_dp = SquareDataPipe(source_dp)

# Iterate through the data pipeline
for data in square_dp:
    print(data)
