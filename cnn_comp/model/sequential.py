"""
Model Container/Neural Network Sequential Model
Network architecture is defined here in a sequential manner.
"""

class SequentialModel:
    """A simple sequential model to stack layers."""
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        """Add a layer to the model."""
        self.layers.append(layer)

    def forward(self, input_data):
        """Forward pass through all layers."""
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output
