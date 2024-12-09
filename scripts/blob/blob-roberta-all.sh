model=blob
backbonetype=roberta-large
eps=0.05
kllr=0.002
beta=0.2
gamma=8

for dataset in rte mrpc wic cola boolq; do
    for sample in 10; do
        for seed in 1 2 3; do
            name=$modelwrapper-$dataset-sample$sample-eps$eps-kllr$kllr-beta$beta-gamma$gamma-seed$seed
            python -m accelerate.commands.launch --num_processes=4 --gpu_ids="0,1,2,3" --main_process_port=29501 run/main.py --dataset-type bertds --dataset $dataset --model $model \
                --model-type seqcls --model $model --modelwrapper $modelwrapper \
                --lr 5e-4 --batch-size 32 \
                --opt adamw --warmup-ratio 0.06 \
                --max-seq-len 256 \
                --seed $seed \
                --wandb-name $name --wandb-project "BLoB-roberta-all" \
                --apply-classhead-lora --lora-r 8 --lora-alpha 8 --lora-dropout 0 \
                --log-path $name \
                --max-train-steps 5000 \
                --eval-per-steps 6000 \
                --bayes-klreweighting \
                --bayes-eps $eps --bayes-beta $beta --bayes-gamma $gamma --bayes-kllr $kllr --bayes-datasetrescaling \
                --bayes-train-n-samples 1 --bayes-eval-n-samples $sample --bayes-eval-n-samples-final $sample
        done
    done
done
