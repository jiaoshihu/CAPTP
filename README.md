# CAPTP

# 1 Description


The identification of *SARS-CoV-2* phosphorylation sites plays important roles in unraveling the complex molecular mechanisms behind infection and the resulting alterations in host cell pathways, thus having profound implications for the fight against the global *COVID-19* pandemic. Here, we developed a novel deep learning predictor called ***PSPred-ALE***, which is specifically designed to identify phosphorylation sites in human host cells that are infected with *SARS-CoV-2*. ***PSPred-ALE*** employs a self-adaptive learning embedding algorithm that automatically extracts sequential features from protein sequences, while also utilizing a multi-head attention module to capture global information, thereby improving the accuracy of predictions. Comparison with benchmarks shows that our proposed predictor, ***PSPred-ALE***, outperforms the current SOTA prediction tools and achieves robust performance. We anticipate that this model will promote the exploration of new phosphorylation modification sites in *SARS-CoV-2* infection and the understanding of the related pathogenesis and therapeutic strategies.


# 2 Requirements

Before running, please make sure the following packages are installed in Python environment:

python==3.8

pytorch==1.13.1

numpy==1.24.2

pandas==1.5.3



# 3 Running

Changing working dir to ATGPred-main, and then running the following command:

python main.py -i test.fasta -o prediction_results.csv

-i: input file in fasta format

-o: output file name
