<img src="docs/source/_static/images/pusion_logo.svg" alt="drawing" width="400"/>

# _pusion_ – Decision Fusion Framework

Pusion (**P**ython **U**niversal Fu**sion**) is a generic and flexible framework written in Python for combining multiple classifier’s decision outcomes.

The framework comprises a variety of decision fusion methods adapted from the field of ensemble techniques and pattern recognition. The general purpose of the framework is to improve classification performance over input classifiers and to determine the compatibility of individual decision fusion methods for a given problem. 
Pusion handles an unlimited number of classifiers and also tolerates various forms of classification output. These include multiclass and multilabel classification, as well as crisp and continuous class assignments (see [documentation](https://ipvs-as.github.io/pusion) for more details).
The framework was originally designed for combining fault detection and fault diagnosis methods, but its application is not limited to this domain.


## Full Documentation
The full HTML documentation is available on [GitHub Pages](https://ipvs-as.github.io/pusion) or in the project's `docs/build/` folder.

Citing
======
If you use the `pusion` framework in your scientific work, please consider citing the related paper:

```
Wilhelm, Y., Reimann, P., Gauchel, W., Klein, S., & Mitschang, B. (2023, April). 
Pusion-A Generic and Automated Framework for Decision Fusion. 
In 2023 IEEE 39th International Conference on Data Engineering (ICDE) (pp. 3282-3295). IEEE.
```

Bibtex:
```
@inproceedings{wilhelm2023pusion,
  title={Pusion-A Generic and Automated Framework for Decision Fusion},
  author={Wilhelm, Yannick and Reimann, Peter and Gauchel, Wolfgang and Klein, Steffen and Mitschang, Bernhard},
  booktitle={2023 IEEE 39th International Conference on Data Engineering (ICDE)},
  pages={3282--3295},
  year={2023},
  organization={IEEE}
}
```
