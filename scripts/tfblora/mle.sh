modelwrapper=tfblora
model=meta-llama/Meta-Llama-3.1-8B
beta=0.015
th=0.003
iter=5
index=0
for dataset in winogrande_s ARC-Challenge ARC-Easy winogrande_m obqa boolq; do
  for sample in 10; do
    for seed in 1 2 3; do
      device=$((index % 3))
      index=$((index + 1))
      name=$modelwrapper-mle-th$th-$dataset-sample$sample-beta$beta-iter$iter-seed$seed
      CUDA_VISIBLE_DEVICES=$device python run/main.py --dataset-type mcdataset --dataset $dataset \
        --model-type causallm --model $model --modelwrapper $modelwrapper \
        --lr 1e-4 --batch-size 50 \
        --opt adamw --warmup-ratio 0.06 \
        --max-seq-len 300 \
        --seed $seed \
        --evaluate \
        --wandb-name $name --wandb-project "var-infer-adaprior-$dataset" \
        --apply-classhead-lora --lora-r 8 --lora-alpha 16 --lora-dropout 0 \
        --log-path $name \
        --max-train-steps 10000 \
        --eval-per-steps 6000 \
        --bayes-klreweighting \
        --load-lora-huggingface-repo FlyLee/bayesian-peft \
        --load-lora-path mle/meta-llama/Meta-Llama-3.1-8B/$dataset/mle-$dataset-sample10-eps0.05-kllr0.0075-beta0.15-seed$seed \
        --testing-set 'train_train_val' \
        --bayes-beta $beta \
        --bayes-train-n-samples $sample --bayes-eval-n-samples $sample --bayes-eval-n-samples-final $sample --th $th --iter $iter &
    done
    wait
  done
done
