# Sample draw the fonts and save to paired_images
PYTHONPATH=. python font2img.py

# Read images, split and convert to numpy arrays, save to pickles
PYTHONPATH=. python package.py --dir=data/paired_images \
                               --save_dir=experiments/data \
                               --split_ratio=0.2

# Train the model
PYTHONPATH=. python train.py --experiment_dir=experiments \
                --experiment_id=0 \
                --batch_size=16 \
                --lr=0.001 \
                --epoch=40 \
                --sample_steps=50 \
                --schedule=20 \
                --L1_penalty=100 \
                --Lconst_penalty=15

# Infer
PYTHONPATH=. python infer.py --model_dir=experiments/checkpoint/experiment_0_batch_16 \
                --batch_size=16 \
                --source_obj=experiments/data/val.obj \
                --embedding_ids=0 \
                --save_dir=save_dir/