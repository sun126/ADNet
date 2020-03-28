<font size=5>**ADNet Implementation using Tensorflow**</font>  

**Requirements**  

1.python  
2.tensorflow  
3.numpy PIL  

**Test**  

python main.py  

    --use_gpu=1 \                           # use gpu or not  
    --gpu_idx=0 \  
    --gpu_mem=0.5 \                         # gpu memory usage  
    --phase=test \  
    --test_dir=/path/to/your/test/dir/ \  
    --save_dir=/path/to/save/results/ \  
    
**Train**  
put your dataset in ./data  
python main.py  

    --use_gpu=1 \                           # use gpu or not  
    --gpu_idx=0 \  
    --gpu_mem=0.5 \                         # gpu memory usage 
    --phase=train \  
    --epoch=100 \                           # number of training epoches  
    --batch_size=16 \  
    --patch_size=48 \                       # size of training patches  
    --start_lr=0.001 \                      # initial learning rate for adm  
    --eval_every_epoch=20 \                 # evaluate and save checkpoints for every # epoches  
    --checkpoint_dir=./checkpoint           # if it is not existed, automatically make dirs  
    --sample_dir=./sample                   # dir for saving evaluation results during training

You can read more details in https://blog.csdn.net/sf_qw39/article/details/105161957  
If you find any problem when running the code, please contact to me.

