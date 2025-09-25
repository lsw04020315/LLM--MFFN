

## Get Start

- Install Python>=3.8, PyTorch 1.8.1.
- Install requirements. pip install -r requirements.txt
- Download data. You can obtain all the benchmarks from [[TimesNet](https://github.com/thuml/Time-Series-Library)].then place the downloaded contents under ./dataset
 
- Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`. 
- You can reproduce the experiment results by:

```bash
bash ./scripts/ETTh1.sh
bash ./scripts/ETTm1.sh
bash ./scripts/weather.sh
bash ./scripts/QBO.sh
```

If you are using Windows 10/11, you can also reproduce the issue by following these steps:

Step 1
Fill in the corresponding hyperparameters[ETTh1.sh,ETTm1.sh,weather.sh and QBO.sh]
Step 2
--task_name long_term_forecast --model_id MFFN-data-seq_len-pred_len
Step 3
run main.py


Acknowledgement
We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/DAMO-DI-ML/ICML2022-FEDformer

https://github.com/thuml/Time-Series-Library

https://github.com/khegazy/One-Fits-All

https://github.com/KimMeen/Time-LLM