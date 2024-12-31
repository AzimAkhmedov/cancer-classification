"""
In this file you may find functions that are used
to calculate statistics and show them
"""
import matplotlib.pyplot as plt


def printEpochs(epochs, loss, val_loss):
    """
    Function that prints the loss of the model
    """
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

