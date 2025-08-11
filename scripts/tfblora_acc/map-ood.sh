modelwrapper=tfblora_acc
model=meta-llama/Meta-Llama-3.1-8B
th=0.003
iter=0
index=0
ori_dataset=obqa

for dataset in ARC-Challenge ARC-Easy MMLU-chem MMLU-phy; do
  for sample in 10; do
    for seed in 1 2 3; do
      if [ "$seed" == "1" ]; then
        beta=0.00625
      elif [ "$seed" == "2" ]; then
        beta=0.00625
      elif [ "$seed" == "3" ]; then
        beta=0.0045000000000000005
      fi
      device=1
      name=$modelwrapper-map-ood-th$th-$dataset-sample$sample-seed$seed
      CUDA_VISIBLE_DEVICES=$device python run/main.py --dataset-type mcdataset --dataset $dataset \
        --model-type causallm --model $model --modelwrapper $modelwrapper \
        --lr 1e-4 --batch-size 50 \
        --opt adamw --warmup-ratio 0.06 \
        --max-seq-len 300 \
        --seed $seed \
        --evaluate \
        --wandb-name $name --wandb-project "$modelwrapper-ood-$dataset" \
        --apply-classhead-lora --lora-r 8 --lora-alpha 16 --lora-dropout 0 \
        --log-path $name \
        --max-train-steps 0 \
        --eval-per-steps 6000 \
        --bayes-klreweighting \
        --load-lora-path /data/local/yibin/bayesian-peft/map/meta-llama/Meta-Llama-3.1-8B/$ori_dataset/map-$ori_dataset-seed$seed \
        --load-model-path /data/local/public_llms/llamas/Meta-Llama-3.1-8B \
        --testing-set 'train_train_val' \
        --bayes-beta $beta \
        --bayes-train-n-samples $sample --bayes-eval-n-samples $sample --bayes-eval-n-samples-final $sample --th $th --iter $iter 
    done
  done
done
