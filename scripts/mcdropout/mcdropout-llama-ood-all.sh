modelwrapper=mcdropout
model=meta-llama/Llama-2-7b-hf
ori_dataset=obqa

for sample in 10; do
    for seed in 1 2 3; do
        name=$modelwrapper-$ori_dataset-seed$seed
        python -m accelerate.commands.launch --num_processes=2 --gpu_ids="0,1" --main_process_port=40000 run/main.py --dataset-type mcdataset --dataset $ori_dataset \
            --model-type causallm --model $model --modelwrapper $modelwrapper \
            --lr 1e-4 --batch-size 4 \
            --opt adamw --warmup-ratio 0.06 \
            --max-seq-len 300 \
            --seed $seed \
            --wandb-name $name --wandb-project "MCD-llama" \
            --apply-classhead-lora --lora-r 8 --lora-alpha 16 --lora-dropout 0.1 \
            --log-path $name \
            --max-train-steps 5000 \
            --eval-per-steps 6000 \
            --bayes-eval-n-samples $sample --bayes-eval-n-samples-final $sample \
            --checkpoint --checkpoint-name $name
    done
done

for dataset in ARC-Challenge ARC-Easy MMLU-chem MMLU-phy; do
    for sample in 10; do
        for seed in 1 2 3; do
            name=$modelwrapper-$dataset-seed$seed
            python -m accelerate.commands.launch --num_processes=2 --gpu_ids="0,1" --main_process_port=40000 run/main.py --dataset-type mcdataset --dataset $dataset \
                --model-type causallm --model $model --modelwrapper $modelwrapper \
                --lr 1e-4 --batch-size 4 \
                --opt adamw --warmup-ratio 0.06 \
                --max-seq-len 300 \
                --seed $seed \
                --wandb-name $name --wandb-project "MCD-llama" \
                --apply-classhead-lora --lora-r 8 --lora-alpha 16 --lora-dropout 0.1 \
                --log-path $name \
                --max-train-steps 0 \
                --eval-per-steps 6000 \
                --bayes-eval-n-samples $sample --bayes-eval-n-samples-final $sample \
                --load-lora-path checkpoints/$modelwrapper/$model/$ori_dataset/$modelwrapper-$ori_dataset-seed$seed 
        done
    done
done
