# Sanskrit Meter Mitra
This repository is created as an initial attempt at creating a Sanskrit poem meter verifier using Chandas (https://github.com/sanskrit-coders/chandas) library. The initial idea was proposed by Rohan Pandey here: https://x.com/khoomeik/status/1925672340819476791

## Setup
The repository has been tested with python=3.11:
```
conda create --prefix=./.conda python=3.11 -y
conda activate ./.conda
pip install -r requirements.txt
```

## Running the example:
```
python reward_metric.py
```
```
Showing demo reward values for different examples:
Text: पूर्णमदः पूर्णमिदं पूर्णात्पूर्णमुदच्यते
पूर्णस्यपूर्णमादाय पूर्णमेवावशिष्यते॥
Meter (mattā): 0.1
Meter (Anuṣṭup (Śloka)): 1.8461877023823723
```

## Reward Function
The reward function `meter_reward_metric(text, meter_name)` in `reward_metric.py` evaluates Sanskrit verse generation quality using this scoring scheme:

### Meter Matching (Primary Criteria)
1. **Exact match**: If the given input text is an exact match for the target meter, then the score is assigned as: 1 + (20 / LMLoss). Here, LMLoss is calculated with a context that contains an example poem in the target meter. This is to assess the naturalness of the generated string. The model currently used is, `google/muril-base-cased`, but open to suggestions!
2. **Partial match**: 0.5
3. **Accidental match**: 0.25
4. **Other exact matches**: 0.1 
5. **Other partial matches**: 0.05
6. **Other accidental matches**: 0.025

### Key Considerations
- 🔄 **Meter canonicalization needed**  
  The current scoring doesn't account for matching a meter similar to the target meter. It can be given a partial reward.

- 🤖 **Naturalness evaluation**  
  Uses cross-entropy loss from a pretrained Sanskrit LM as text fluency proxy. I would be glad to receive suggestions to improve this, either via a better model, or different mechanism like LLM-as-a-judge.
