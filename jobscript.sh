                                               
#!/bin/sh
### -------- specify queue name ----------------
#BSUB -q c02516

### -------- specify gpu request ---------------
#BSUB -gpu "num=1:mode=exclusive_process"

### -------- specify job name ------------------
#BSUB -J testrun_singleframe

### -------- specify number of cores -----------
#BSUB -n 4
#BSUB -R "span[hosts=1]"

### -------- specify CPU memory ----------------
#BSUB -R "rusage[mem=20GB]"

### -------- specify wall-clock time -----------
#BSUB -W 01:30   # 30 minutter

### -------- specify output/error files --------
#BSUB -o logs/output_%J.out
#BSUB -e logs/error_%J.err


### -------- execution environment -------------
source ~/venv_1/bin/activate

### -------- run your script -------------------
python -m src.main
