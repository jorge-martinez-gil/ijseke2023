(repository under construction)

# A Comparative Study of Ensemble Techniques Based on Genetic Programming: A Case Study in Semantic Similarity Assessment
Code of the paper *J. Martinez-Gil: A Comparative Study of Ensemble Techniques Based on Genetic Programming: A Case Study in Semantic Similarity Assessment. Int. J. Softw. Eng. Knowl. Eng. 33(2): 289-312 (2023)*

[https://doi.org/10.1142/S0218194022500772](https://doi.org/10.1142/S0218194022500772 "https://doi.org/10.1142/S0218194022500772")

# Synopsis
Semantic similarity assessment is an important task in NLP, which aims to measure the degree of similarity between two pieces of text. There are various techniques that have been developed to solve this problem, including our ensemble techniques based on genetic programming. We present a comparative study of different ensemble techniques based on genetic programming for semantic similarity assessment.

# Methodology
This repository explores the application of genetic programming to the problem of semantic similarity assessment. The paper compares various ensemble techniques that use genetic programming to combine multiple similarity measures in order to improve the accuracy of semantic similarity assessments. We perform experiments using several datasets and find that the ensemble techniques outperform individual similarity measures, demonstrating the potential of genetic programming in the field of natural language processing.

# Dependencies
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

# Usage

- (BASELINE) Linear Regression (LR)

- Linear Genetic Programming (LGP)

- Tree Genetic Programming (TGP)

- Cartesian Genetic Programming (CGP)


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