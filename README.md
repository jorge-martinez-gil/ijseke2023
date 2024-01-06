
<!-- Add Attractive Title with Emojis -->
<h1 align="center">
 A Comparative Study of Ensemble Techniques Based on Genetic Programming 
</h1>

*Code for reproducing:*

J. Martinez-Gil: A Comparative Study of Ensemble Techniques Based on Genetic Programming: A Case Study in Semantic Similarity Assessment. Int. J. Softw. Eng. Knowl. Eng. 33(2): 289-312 (2023)

<!-- Add DOI Badge -->
<p align="center">
  <a href="https://doi.org/10.1142/S0218194022500772">
    <img src="https://img.shields.io/badge/DOI-10.1142%2FS0218194022500772-blue" alt="DOI">
  </a>
</p>

<!-- Add Synopsis Section -->
## :book: Synopsis
Semantic similarity assessment is a crucial task in Natural Language Processing (NLP), aiming to quantify text similarity. In this research, we delve into cutting-edge ensemble techniques based on genetic programming, designed to enhance semantic similarity assessment. Join us on an exciting journey through the world of NLP and genetic programming.

<!-- Add Methodology Section -->
## :rocket: Methodology
This repository is a treasure trove of knowledge on harnessing genetic programming for semantic similarity assessment. Our paper meticulously evaluates diverse ensemble techniques employing genetic programming to fuse multiple similarity measures. The results reveal their potential to elevate the accuracy of semantic similarity assessments, paving the way for innovations in NLP.

<!-- Add Methods with Emojis -->
### :microscope: Tested Methods
The following methods are tested:
- **(BASELINE) Linear Regression (LR)**
  - :snake: (Python) A simple statistical method to model the relationship between variables. It predicts similarity based on linear relationships between text features.
- **Linear Genetic Programming (LGP)**
  - :coffee: (Java) An evolutionary technique using computer programs as solutions. It evolves linear programs to aggregate similarity measures for better predictions.
- **Tree Genetic Programming (TGP)**
  - :snake: (Python) Similar to LGP, but represents programs as trees. It evolves hierarchical programs for aggregating multiple similarity measures.
- **Cartesian Genetic Programming (CGP)**
  - :snake: (Python) It is a genetic programming variant with fixed grid-like structures for efficient program representation. It evolves connections and functions for semantic similarity assessment.

<!-- Add Dependencies Section -->
## :gear: Dependencies

### For LR (Python)
```python
gplearn==0.4.2
numpy==1.25.2
pandas==1.1.5
scipy==1.9.1
```

### For LGP (java)
```java
import com.github.chen0040.gp.lgp.LGP;
import com.github.chen0040.data.utils.TupleTwo;
import com.github.chen0040.gp.commons.BasicObservation;
import com.github.chen0040.gp.commons.Observation;
import com.github.chen0040.gp.lgp.gp.Population;
import com.github.chen0040.gp.lgp.program.Program;
import com.github.chen0040.gp.lgp.program.operators.*;
import com.github.chen0040.gp.services.Tutorials;
import com.github.chen0040.gp.utils.CollectionUtils;
```
### For TGP (python)
```python
gplearn==0.4.2
numpy==1.25.2
pandas==1.1.5
scipy==1.9.1
```
### For CGP (python)
```python
numpy==1.25.2
pandas==1.1.5
scikit_learn==1.3.0
scipy==1.9.1
tengp==0.4
```

# Usage
Before running each program, be sure to define the target (Pearson or Spearman Rank) as well as the parameters for the grid search. Keep in mind that underpowered computers will not be very efficient at running a search with a large solution space. 

- **LR**.  `python lr.py`
- **LGP**. Install the Eclipse project, and run lgp.main
- **TGP**. `python tgp.py`
- **CGP**. `python cgp.py`

The most important thing is that all methods use the same training and test datasets to ensure the fairest possible comparison.

# Citation
If you use this work, please cite:
```
@article{martinezgil2023c,
  author       = {Jorge Martinez-Gil},
  title        = {A Comparative Study of Ensemble Techniques Based on Genetic Programming: {A} Case Study in Semantic Similarity Assessment},
  journal      = {Int. J. Softw. Eng. Knowl. Eng.},
  volume       = {33},
  number       = {2},
  pages        = {289--312},
  year         = {2023},
  url          = {https://doi.org/10.1142/S0218194022500772},
  doi          = {10.1142/S0218194022500772}
}
```

# License
This code is released under the MIT License. See the LICENSE file for more information.