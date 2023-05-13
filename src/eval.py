def evaluate(predictions, targets):
    """
    Computes accuracy given predicted and target labels
    input:
        predictions: a list of predicted labels
        targets: a list of true labels
    output:
        accuracy: float value of the classification accuracy
    """
    correct = 0
    total = len(predictions)

    for i in range(total):
        if predictions[i] == targets[i]:
            correct += 1
    accuracy = correct / total

    # To do : significance measure
    # Save results somewhere

    return accuracy
