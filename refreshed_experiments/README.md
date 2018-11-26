Basic setup for running classification experiments.
You can run them with `run.py`. Example usage:
```
python run.py -mc convgp_creator -d svhn_5percent -c TickConvGPConfig -t ClassificationGPTrainer -p /tmp/results
```
This will build the model using `convgp_creator` using `TickConvGPConfig` config and run 
`ClassificationGPTrainer` on `svhn_5percent` dataset. The results will be stored to `/tmp/results`. 
Run python run.py --help to see more detailed description. 

Then you can analyse the results by running:
```
python analyse.py -r /tmp/results -s loss acc
```
This will gather all the results stored in `/tmp/results` and print a summary string.
