import torch
from torchsummary import summary

class ModelInspector:
    def __init__(self, model_class, checkpoint_path, device, input_size):
        """
        Initialize the ModelInspector with a given model class, checkpoint path, and device.

        Parameters:
        - model_class: The class of the model (e.g., a subclass of nn.Module).
        - checkpoint_path: The path to the .pth model checkpoint file.
        - device: The device to load the model on (e.g., 'cuda' or 'cpu').
        """
        self.model_class = model_class
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = self._load_model()
        self.input_size = input_size

    def _load_model(self):
        """
        Load the model from the checkpoint.

        Returns:
        - model: The loaded PyTorch model.
        """
        model = self.model_class().to(self.device)
        model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        model.eval()  # Set the model to evaluation mode by default
        return model

    def inspect_model(self):
        """
        Print a summary of the model architecture.

        Parameters:
        - input_size: A tuple representing the input size of the model (e.g., (3, 224, 224) for an image model).
        """

        print("Model Summary:")
        summary(self.model, self.input_size)

    def make_layers_trainable(self, layers_to_train):
        """
        Make specific layers trainable in the model.

        Parameters:
        - layers_to_train: A list of layer names (as strings) that you want to make trainable.
        """
        # Set all layers to non-trainable initially
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Make specified layers trainable
        for name, param in self.model.named_parameters():
            for layer in layers_to_train:
                if layer in name:
                    param.requires_grad = True
                    print(f'Layer "{name}" is now trainable.')

        self.inspect_model()

        return self.model


    def print_layer_names(self):
        """
        Print the names of all layers in the model.
        """
        print("Layer names in the model:")
        for name, param in self.model.named_parameters():
            print(name)
