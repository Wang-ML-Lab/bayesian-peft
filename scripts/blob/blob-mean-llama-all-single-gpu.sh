modelwrapper=blob
model=meta-llama/Llama-2-7b-hf
eps=0.05
beta=0.2
kllr=0.01
gamma=8

for dataset in winogrande_s ARC-Challenge ARC-Easy winogrande_m obqa boolq; do
    for sample in 1; do
        for seed in 1 2 3; do
            name=$modelwrapper-$dataset-sample$sample-eps$eps-kllr$kllr-beta$beta-gamma$gamma-seed$seed
            CUDA_VISIBLE_DEVICES=0 python run/main.py --dataset-type mcdataset --dataset $dataset \
                --model-type causallm --model $model --modelwrapper $modelwrapper \
                --lr 1e-4 --batch-size 4 \
                --opt adamw --warmup-ratio 0.06 \
                --max-seq-len 300 \
                --seed $seed \
                --evaluate \
                --wandb-name $name --wandb-project "BLoB-llama-all" \
                --apply-classhead-lora --lora-r 8 --lora-alpha 16 --lora-dropout 0 \
                --log-path $name \
                --max-train-steps 5000 \
                --eval-per-steps 6000 \
                --bayes-klreweighting \
                --bayes-eps $eps --bayes-beta $beta --bayes-gamma $gamma --bayes-kllr $kllr --bayes-datasetrescaling \
                --bayes-inference-notsample \
                --bayes-train-n-samples 1 --bayes-eval-n-samples 1 --bayes-eval-n-samples-final $sample 
        done
    done
done


