set -ex
for MODEL in char_rnn vae aae
do
    for SEED in 1 2 3
    do
        mkdir -p checkpoints/$MODEL/$MODEL\_$SEED
        python scripts/run.py \
            --model $MODEL \
            --data data \
            --checkpoint_dir checkpoints/$MODEL/$MODEL\_$SEED \
            --device cuda:$SEED \
            --metrics data/samples/$MODEL/metrics_$MODEL\_$SEED.csv \
            --seed $SEED \
            --gen_path data/samples/$MODEL/$MODEL\_$SEED.csv &
    done
done
