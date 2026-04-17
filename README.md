<div align="center">
<h1> Beyond Text-Dominance: Understanding Modality Preference of Omni-modal Large Language Models


[![arxiv](] [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) 
</div>
 
[stars-img]: https://img.shields.io/github/stars/QingFenwy7/omni_modality_preference?color=yellow
[stars-url]: https://github.com/QingFenwy7/omni_modality_preference/stargazers
[fork-img]: https://img.shields.io/github/forks/QingFenwy7/omni_modality_preference?color=lightblue&label=fork
[fork-url]: https://github.com/QingFenwy7/omni_modality_preference/network/members

<div align="center">
<h1>[![GitHub stars][stars-img]][stars-url]
<h1> [![GitHub forks][fork-img]][fork-url]

-------------
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
📍 **Data**:
<p align = "justify"> 
 MSR (%) results of all evaluated OLLMs on the tri-modal conflict dataset. 
</p>
<div  align="center">    
    <img src="./assets/1.1.png" width=60%/>
</div>


### Preference Emerges
<p align = "justify"> 
Layer-wise modality preference probe accuracy for all evaluated OLLMs.
</p>
<div  align="center">    
    <img src="./assets/2.1.png" width=60%/>
</div>


### Hallucination Detection
<p align = "justify"> 
Case study.
</p>
<div  align="center">    
    <img src="./assets/3.1.png" width=90%/>
</div>



## 📖 Citation

If you find this project helpful, please use the following to cite it:

```

```
