This code repository implements the experiments described in "[Approximating CKY with Transformers](https://arxiv.org/abs/2305.02386)" by Ghazal Khalighinejad, Ollie Liu, and Sam Wiseman, to appear in findings of EMNLP 2023.

This code includes the implementation of constituency parsing and synthetic experiments. The code for constituency parsing experiments adapts the [self-attentive-parser](https://github.com/nikitakit/self-attentive-parser) by [Kitaev et al.](https://arxiv.org/abs/1812.11760). 

### Making the Synthetic Data
For generating data from PCFG with 100 rules:
```
python3 -u make_data.py --cpu --nsamples 200000 --data-path data/pcfgs/term5k-nt20-birul100-lexrul5k-3 --save data/birul100-200k/data
```
For generating data from PCFG with 400 rules:
```
python3 -u make_data.py --cpu --nsamples 200000 --data-path data/pcfgs/term5k-nt20-birul400-lexrul5k-3 --save data/birul400-200k/data
```
For generating data from PCFG with 800 rules:
```
python3 -u make_data.py --cpu --nsamples 200000 --data-path data/pcfgs/term5k-nt20-birul800-lexrul5k-3 --save data/birul800-200k/data
```
### Synthetic Experiments
```
python3 -u main.py --from-scratch --multiple-files --data-path get_data/data/birul100-200k --clip 1 --learning-rate 0.00010 --train-batch-size 32 --attention-probs-dropout-prob 0.3 --hidden-dropout-prob 0.3 --warmup-steps 4000 --weight-decay 0.001 
```
```
python3 -u main.py --from-scratch --multiple-files --data-path get_data/data/birul400-200k --clip 1 --learning-rate 0.00010 --train-batch-size 32 --attention-probs-dropout-prob 0.3 --hidden-dropout-prob 0.3 --warmup-steps 4000 --weight-decay 0.001
```
```
python3 -u main.py --from-scratch --multiple-files --data-path get_data/data/birul800-200k --clip 1 --learning-rate 0.00013 --train-batch-size 32 --attention-probs-dropout-prob 0.3 --hidden-dropout-prob 0.3 --warmup-steps 4000 --weight-decay 0.001 
```


