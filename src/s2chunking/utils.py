def validate_data(data):
    """Validate the input data."""
    if not isinstance(data, (list, str)):
        raise ValueError("Data must be a list or a string.")
    return True

def format_chunk(chunk):
    """Format the output chunk."""
    return str(chunk)  # Convert chunk to string for consistent formatting