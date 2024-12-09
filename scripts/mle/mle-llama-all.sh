modelwrapper=mle
model=meta-llama/Llama-2-7b-hf

for dataset in winogrande_m; do
    for seed in 1 2 3; do
        name=$modelwrapper-$dataset-seed$seed
        python -m accelerate.commands.launch --num_processes=2 --gpu_ids="0,1" --main_process_port=20000 run/main.py --dataset-type mcdataset --dataset $dataset \
            --model-type causallm --model $model --modelwrapper $modelwrapper \
            --lr 1e-4 --batch-size 4 \
            --opt adamw --warmup-ratio 0.06 \
            --max-seq-len 300 \
            --seed $seed \
            --wandb-name $name --wandb-project "MLE-llama-all" \
            --apply-classhead-lora --lora-r 8 --lora-alpha 16 --lora-dropout 0 \
            --log-path $name \
            --max-train-steps 5000 \
            --eval-per-steps 6000
    done
done
