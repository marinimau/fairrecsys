# FairRecSys

### Brief Description

FairRecSys is a post processing method in order to have less discriminative recommendations.

### Requirements
- Python
- google ortools
- pandas
- numpy
- sklearn
- scipy

### Usage
To run algorithm with metric \mu_1:
```sh
$ python alg_distMet_mu1.py \
altered_matrix_path \
userFile_path \
predictions_path \ 
topK \
epsilon
```

To run algorithm with metric \mu_2:
```sh
$ python alg_distMet_mu2.py \
altered_matrix_path \
userFile_path \
predictions_path \ 
topK \
epsilon
```

Explanation for Input parameters:
"altered_matrix_path": path to output altered matrix
"userFile_path": path to user file where sensitive attribute exists 
"predictions_path": original recommendation matrix
"topK": top-k
"epsilon": desired level of fairness

Example: 
```sh 
$ python alg_distMet_mu1.py altered_fair_matrix ../data/ml-1m-users.csv ../data/ml-1m-pred-wrmf.csv 20 0.2
```

### Questions
bedizel@gmail.com
