#Dict-NN

## Code for integrated Dictionary Learning+Neural Networks model introduced in https://link.springer.com/chapter/10.1007/978-3-030-32248-9_79
![Dict-NN](https://github.com/Niharika-SD/Dict-NN/Connectomics_and_Clinical_Severity_NN.PNG)

#INSTRUCTIONS

Open terminal, type: python main.py

#DATA ORGANIZATION

Main directory contents:
 1. main.py - main script
 2. Alternating_Minimization.py - scripts for alternating minimization procedure
 3. Optimization_Modules.py - scripts for individual optimization steps 
 4. Helpers.py - scripts for general processing and data aggregation
 5. Quadratic_Solver.py - QP solver required at test time
 6. ANN.py - ANN definition script

~/Data/ contents

  ~/Data/data.mat #dataset
  ~/Data/Outputs/Loss.png #Loss curve
  ~/Data/Outputs/B_gd.png #Basis
  ~/Data/Outputs/Performance.mat #regression performance
  ~/Data/Outputs/deep_sr-DDL.p #saved models
  ~/Data/Outputs/logfile.txt #logs
      
   
