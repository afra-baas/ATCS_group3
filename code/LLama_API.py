
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")

# prepare input
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')

# forward pass
output = model(**encoded_input)
print(output)







# ___________________________________________________
# from transformers import LlamaForCausalLM, LlamaTokenizer

# tokenizer = LlamaTokenizer.from_pretrained("/output/path")
# model = LlamaForCausalLM.from_pretrained("/output/path")


# from transformers import LlamaModel, LlamaConfig

# # Initializing a LLaMA llama-7b style configuration
# configuration = LlamaConfig()

# # Initializing a model from the llama-7b style configuration
# model = LlamaModel(configuration)

# # Accessing the model configuration
# configuration = model.config
