# API Documentation for S2 Chunking Lib

## Overview

The S2 Chunking Lib is a Python library designed to facilitate the chunking of data into manageable pieces. This documentation provides an overview of the classes and methods available in the library.

## Classes

### Chunker

The `Chunker` class is the main class responsible for chunking data.

#### Methods

- **`__init__()`**
  - Initializes a new instance of the `Chunker` class.
  
- **`set_chunk_size(size: int)`**
  - Sets the size of each chunk.
  - **Parameters:**
    - `size` (int): The desired size of each chunk.
  
- **`chunk_data(data)`**
  - Takes input data and returns it in chunks.
  - **Parameters:**
    - `data` (iterable): The input data to be chunked.
  - **Returns:**
    - An iterable of chunks.

## Usage Example

```python
from s2chunking.chunker import Chunker

chunker = Chunker()
chunker.set_chunk_size(5)
data = range(20)
chunks = chunker.chunk_data(data)

for chunk in chunks:
    print(chunk)
```

## Utility Functions

### validate_data(data)

Checks the validity of the input data.

### format_chunk(chunk)

Formats the output chunk for better readability.