def label_mapping(labels, map):
    """
    Maps labels from data loader to the desired labels.
    input:
        labels: list of true labels
        map: a dict where each key represents a true label and each value is the mapping value
    output:
        output: a list of mapped labels.
    """
    output = []
    for label in labels:
        if label not in map:
            print(f"couldn't find {label} in the mapping")
            output.append(None)
        else:
            output.append(map[label])
    return output