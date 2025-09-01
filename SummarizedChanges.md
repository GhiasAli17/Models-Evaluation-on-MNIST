# Summary of Changes from Previous Version  

The following summarizes the key updates and improvements made compared to the previous version:  

1. Added stratified splitting for uniform distribution in the validation set.  
2. Added CNN architecture with dropout, learning rate, and weight decay, improving generalization and stabilizing convergence.  
3. Increased the number of epochs from 30 to 50, leading to higher accuracy:    
   - MLP: 97.93% → 98.12%    
   - CNN: 98.77% → 99.07% (TensorFlow implementation)    

| Model | #Params | Training Time (min) | Test Acc % | Macro-F1 | Avg time per epoch (min) |  
|-------|---------|-------------------|------------|----------|-------------------------|  
| ML    | N/A     | 1.06              | 97.42      | 0.97     | N/A                     |  
| MLP   | 235,914 | 3.39              | 98.12      | 0.98     | 0.13                    |  
| CNN   | 1,219,614 | 383.85          | 99.07      | 0.99     | 7.68                    |  

4. Reimplemented the models in PyTorch using a modular architecture for both MLP and CNN.    
5. Updated the results section to include accuracy, macro-F1 score, and per-epoch training times. 
