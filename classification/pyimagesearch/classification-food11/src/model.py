from torch.cuda.amp import autocast
from torch import nn

class FoodClassifier(nn.Module):
    def __init__(self, base_model, num_classes) -> None:
        super(FoodClassifier, self).__init__()

        # Initialize the base model and classifier layer
        self.base_model = base_model
        self.classifier = nn.Linear(base_model.classifier.in_features, num_classes)

        # Since we are using our own classification layer, we replace the inbuild classifier layer of the base model with nn.Identity which is nothing but a placeholder layer
        # Hence set the classifier of our base model to produce outputs from the last convolution block
        self.base_model.classifier = nn.Identity()

    
    # We decorate the *forward()* method with *autocast()* to enable mixed precision training in a distributed manner
    # which is essentially makes our training faster due to the smart assignment of datatypes
    @autocast()
    def forward(self, x):
        # Pass the input through basemodel and then obtain the classifier outputs
        features = self.base_model(x)
        logits = self.classifier(features)

        # Return the classifier outputs
        return logits