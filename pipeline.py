from prompt_generator import prompt_generator
from label_mapping import label_mapping
def pipeline(Datasetpath, prompt_instructions, prompt_querry, label_map, LM_model):

    #ToDo Dataloader
    #sentences, labels = Dataloader(datasetpath)

    #Generate promts
    prompts = prompt_generator(prompt_instructions, prompt_querry, sentences)

    #map labels
    mapped_labels = label_mapping(labels, label_map)

    #ToDo classificaton
    #logits = Classifier(prompt, list(label_map.keys()), LM_model)

    #ToDo Evaluation:
    #acc = eval(logits, mapped_labels)

    return acc
