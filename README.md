# W-Pathsim

This is the source code of *W-PathSim* which is a similarity measure algorithm for the content-based heterogeneous information network (C-HIN). In this work, we formally defined a content-based meta-path-based similarity measurement to leverage the quality of multi-typed node similarity search in the complex C-HINs. 

## Requirements
- Python >= 3.6
- NetworkX >= 2.3
- Scipy >= 1.3
- Numpy >= 1.17

## Dataset usage
- The DBLP bibliographic network (https://dblp.uni-trier.de/) is the main experimental dataset. 
- The topic distributions of papers are extracted from the papers' abstracts (retrieved from Aminer repository: https://www.aminer.org/) by using the LDA topic modelling of the Scikit-Learn library (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html) 

## Citing
If you find W-PathSim algorithm is useful in your researches, please cite the following paper:

    @inproceedings{pham2018w,
      title={W-PathSim: novel approach of weighted similarity measure in content-based heterogeneous information networks by applying LDA topic modeling},
      author={Pham, Phu and Do, Phuc and Ta, Chien DC},
      booktitle={Asian conference on intelligent information and database systems},
      pages={539--549},
      year={2018},
      organization={Springer}
    }

## Miscellaneous

Please send any question you might have about the code and/or the algorithm to <phamtheanhphu@gmail.com>.