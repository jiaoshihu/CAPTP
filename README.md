# CAPTP

# 1 Description


Peptides offer significant therapeutic potential for treating a wide range of diseases due to their specificity and efficacy. However, the development of peptide-based drugs is often hindered by the potential toxicity of peptides, which poses a significant barrier to their clinical application. Traditional experimental methods for evaluating peptide toxicity are time-consuming and costly, making the development process inefficient. Therefore, there is an urgent need for computational tools specifically designed to predict peptide toxicity accurately and rapidly, facilitating the identification of safe peptide candidates for drug development. We provide here a novel computational approach, ***CAPTP***, to predict peptide toxicity directly from amino acid sequences. Compared with other models, our method demonstrates better performance, including the absence of overfitting, strong robustness and powerful generalizability. ***CAPTP*** not only identifies potentially toxic peptides with high accuracy but also provides insights into sequential patterns that contribute to toxicity, guiding the design of safer peptide drugs.



# 2 Requirements

Before running, please make sure the following packages are installed in Python environment:

python==3.8.16

pytorch==2..0

numpy==1.24.3

pandas==2.0.1



# 3 Running

Changing working dir to ATGPred-main, and then running the following command:

python main.py -i test.fasta -o prediction_results.csv

-i: input file in fasta format

-o: output file name
