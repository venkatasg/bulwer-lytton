## Overview

This repository contains the complete implementation, data, and analysis for "*Dark & Stormy*: Modeling Humor in the Worst Sentences Ever Written", a research paper that investigates how *intentionally bad* textual humor differs from standard humor datasets. The paper curates and analyzes a novel corpus of 1,778 sentences from the **[Bulwer-Lytton Fiction Contest](https://www.bulwer-lytton.com) (BLFC)** (winners and highlighted entries from 1996-2024), compares them with LLM-generated sentences, and examines the linguistic features that distinguish this unique form of humor.

## Key Findings

- **Humor Detection Failure**: Standard RoBERTa-based humor detection models perform poorly on BLFC sentences, with humor scores comparable to non-humorous novel opening sentences.
- **Distinct Literary Devices**: BLFC sentences disproportionately use irony, metafiction, metaphor, and simile compared to standard humor datasets.
- **Novel Adjective-Noun Expressions**: BLFC sentences contain significantly more semantically deviant (novel) adjective-noun bigrams (10% of BLFC ANs occur <100 times in DCLM corpus vs. 4-6% in baselines).
- **LLM Exaggeration**: While LLMs can imitate the form of BLFC humor, they consistently exaggerate its features—particularly metaphor, simile, and number of novel adjective-noun bigrams.
- **Surprisal Patterns**: Unlike standard humor datasets with concentrated surprisal at sentence boundaries, BLFC sentences exhibit flatter distributions of high-surprisal tokens

## Repository Structure

```
bulwer-lytton/
├── Bulwer_Lytton.pdf                      # The research paper
├── README.md                              # This file
├── LICENSE                                # MIT License
├── data/                                  # Data files
│   ├── Bulwer-Lytton.tsv                  # 1,778 human-written BLFC entries 
│   ├── *_LLM_*.txt                        # Synthetic sentences from various LLMs
│   └── README.md                          # Dataset documentation
├── humor_detection/                       # Humor detection model experiments
├── literary_device_analysis/              # Literary device extraction & analysis
├── oneshot-generation/                    # LLM sentence generation
├── adjective-noun/                        # Adjective-noun analysis
└── Surprisal.ipynb                        # Token surprisal analysis
```

## Paper Reference

**Citation:**
```
```

## Limitations

- Focus on a single form of intentionally bad humor (BLFC entries).
- Humor detection models may not fully capture humor complexity.
- Literary device analysis limited to 8 features identified by GPT-4.1
- Surprisal analysis doesn't directly imply humor location without human annotation.
- LLM evaluation focuses on stylistic differences, not subjective quality judgment.


## Contact

- **Venkata S Govindarajan**: vgovindarajan@ithaca.edu (Ithaca College)
- **Laura Biester**: lbiester@middlebury.edu (Middlebury College)