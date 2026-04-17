<div align="center">
<h1> Beyond Text-Dominance: Understanding Modality Preference of Omni-modal Large Language Models </h1>


[![arxiv](] [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) 
</div>



## 🎯 What this paper does
* **Proposing a modality preference evaluation framework for OLLMs:** Constructing a tri-modal semantic conflict dataset with quantitative metrics to systematically measure model modality preferences.
* **Revealing the modality preference landscape of OLLMs:** Under tri-modal conflicts, most OLLMs exhibit significant visual preference; under bi-modal conflicts, all models favor the visual modality; across all input combinations, the audio modality is systematically neglected.
* **Revealing the internal evolution patterns of modality preference:** Employing layer-wise linear probing to reveal that modality preference signals are absent in shallow layers and gradually emerge in mid-to-late layers.
* **Leveraging linear probes for hallucination detection:** Discovering that hallucination generation is accompanied by abnormally elevated preference probability toward the interfering modality, enabling effective hallucination detection via linear probes.


## 🔮 Usage
📍 **Data**:
```
data/conflict_triplets_processed.json
```
📍 **Eval**:
```
eval/run_tri-modal.py
```
📍 **Probe**:
```
probe/train.py   # Train linear probes
probe/acc.py     # Calculate accuracy
probe/pred.py    # Predict preference probabilities
```



## 📖 Citation

If you find this project helpful, please use the following to cite it:

```

```
