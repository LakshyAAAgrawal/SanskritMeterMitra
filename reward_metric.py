from chandas import identifier
from chandas import to_pattern_lines
from chandas import svat_identifier

from chandas.svat.data import curated, dhaval_mishra, dhaval_vrttaratnakara, ganesh

all_meters_list = set([
    x[0] for x in 
    curated.curated_vrtta_data + dhaval_mishra.dhaval_vrtta_data + dhaval_vrttaratnakara.data_vrttaratnakara + ganesh.data
])

# Global cache for models/tokenizers
_model_cache = {}

def calculate_loss(model_name, input_text, input_key, only_text=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    from meter_examples import meter_examples
    
    # Load model/tokenizer if not cached
    if model_name not in _model_cache:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        _model_cache[model_name] = (model, tokenizer)
    
    model, tokenizer = _model_cache[model_name]
    
    # Build context using example mapping
    example = meter_examples.get(input_key, "")
    if example == "":
        context = f"An example of a sanskrit poem in the meter {input_key} is: "
    else:
        context = f"Some examples of a sanskrit poem in the meter {input_key} are:\n1. {example}\n2. "

    # # Tokenize and calculate loss
    # inputs = tokenizer(context, return_tensors="pt")
    # with torch.no_grad():
    #     outputs = model(**inputs, labels=inputs["input_ids"])
    
    # return outputs.loss.item()

    # Concatenate context and input_text
    full_input = context + input_text

    # Tokenize context and full input together
    context_ids = tokenizer(context, return_tensors="pt")
    full_input_ids = tokenizer(full_input, return_tensors="pt")

    context_len = context_ids["input_ids"].size(1)
    input_len = full_input_ids["input_ids"].size(1)

    # Prepare labels: -100 for context, normal for input_text
    labels = full_input_ids["input_ids"].clone()
    if only_text:
        labels[:, :context_len] = -100

    with torch.no_grad():
        outputs = model(**full_input_ids, labels=labels)

    raw_loss = outputs.loss.item()
    # Normalize so 1 = best, 0 = worst
    # normalized_score = 1 / (1 + (raw_loss/20))
    
    return raw_loss

def meter_reward_metric(text, meter_name):
    # TODO: There are many cases where meters names are synonyms or closely related. 
    # Need to canonicalize the meter names for identical, and at least give a high partial score for closely related.
    # For example: pāñcālāṅghriḥ and gajagatiḥ
    # assert meter_name in all_meters_list, f"Meter {meter_name} not found in all_meters_list"
    pattern_lines = to_pattern_lines(text.split("\n"))
    o = svat_identifier.IdentifyFromPatternLines(pattern_lines)

    score = 0

    if "exact" in o and meter_name in o['exact']:
        score += 1
        # Another possible model to try: "buddhist-nlp/buddhist-sentence-similarity"
        # TODO: This is a temporary hack to get a score. Need to find a better way to do this.
        score += 20 / calculate_loss("google/muril-base-cased", text, meter_name)
    elif "partial" in o and meter_name in o['partial']:
        score += 0.5
    elif "accidental" in o and meter_name in o['accidental']:
        score += 0.25
    elif "exact" in o and len(o['exact']) > 0:
        score += 0.1
    elif "partial" in o and len(o['partial']) > 0:
        score += 0.05
    elif "accidental" in o and len(o['accidental']) > 0:
        score += 0.025
    
    return score

if __name__ == "__main__":
    print("Showing demo reward values for different examples:")
    text = """पूर्णमदः पूर्णमिदं पूर्णात्पूर्णमुदच्यते
पूर्णस्यपूर्णमादाय पूर्णमेवावशिष्यते॥"""
    print("Text:", text)
    print("Meter (mattā):", meter_reward_metric(text, "mattā"))
    print("Meter (Anuṣṭup (Śloka)):", meter_reward_metric(text, 'Anuṣṭup (Śloka)'))
    print("===========")
