export CUDA_VISIBLE_DEVICES=0

mkdir -p different_lengths_logs

################################ Full Attention ################################
for bsz in 1 2 4 8
do
    for round in 1
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type Full_Flash_Attn \
            --context_len 60000 \
            --task_name NIAH \
            --batch_size $bsz > different_lengths_logs/full_attn_60k_bsz${bsz}_${round}.log 2>&1
    done
done


for bsz in 1 2 4
do
    for round in 1
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type Full_Flash_Attn \
            --context_len 120000 \
            --task_name NIAH \
            --batch_size $bsz > different_lengths_logs/full_attn_120k_bsz${bsz}_${round}.log 2>&1
    done
done


for bsz in 1 2
do
    for round in 1
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type Full_Flash_Attn \
            --context_len 240000 \
            --task_name NIAH \
            --batch_size $bsz > different_lengths_logs/full_attn_240k_bsz${bsz}_${round}.log 2>&1
    done
done


for bsz in 1
do
    for round in 1
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type Full_Flash_Attn \
            --context_len 480000 \
            --task_name NIAH \
            --batch_size $bsz > different_lengths_logs/full_attn_480k_bsz${bsz}_${round}.log 2>&1
    done
done


################################ RetroInfer ################################
# 60K
for bsz in 1 2 4 8 16 32
do
    for round in 1
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --context_len 60000 \
            --task_name NIAH \
            --batch_size $bsz > different_lengths_logs/retroinfer_60k_bsz${bsz}_${round}.log 2>&1
    done
done

for bsz in 64
do
    for round in 1
    do
        numactl --cpunodebind=0 --membind=0,1 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --context_len 60000 \
            --task_name NIAH \
            --batch_size $bsz > different_lengths_logs/retroinfer_60k_bsz${bsz}_${round}.log 2>&1
    done
done

# 120K
for bsz in 1 2 4 8 16
do
    for round in 1
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --context_len 120000 \
            --task_name NIAH \
            --batch_size $bsz > different_lengths_logs/retroinfer_120k_bsz${bsz}_${round}.log 2>&1
    done
done

for bsz in 32
do
    for round in 1
    do
        numactl --cpunodebind=0 --membind=0,1 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --context_len 120000 \
            --task_name NIAH \
            --batch_size $bsz > different_lengths_logs/retroinfer_120k_bsz${bsz}_${round}.log 2>&1
    done
done

# 240K
for bsz in 1 2 4 8
do
    for round in 1
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --context_len 240000 \
            --task_name NIAH \
            --batch_size $bsz > different_lengths_logs/retroinfer_240k_bsz${bsz}_${round}.log 2>&1
    done
done

for bsz in 16
do
    for round in 1
    do
        numactl --cpunodebind=0 --membind=0,1 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --context_len 240000 \
            --task_name NIAH \
            --batch_size $bsz > different_lengths_logs/retroinfer_240k_bsz${bsz}_${round}.log 2>&1
    done
done

# 480K
for bsz in 1 2 4
do
    for round in 1
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --context_len 480000 \
            --task_name NIAH \
            --batch_size $bsz > different_lengths_logs/retroinfer_480k_bsz${bsz}_${round}.log 2>&1
    done
done

for bsz in 8
do
    for round in 1
    do
        numactl --cpunodebind=0 --membind=0,1 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --context_len 480000 \
            --task_name NIAH \
            --batch_size $bsz > different_lengths_logs/retroinfer_480k_bsz${bsz}_${round}.log 2>&1
    done
done

# 1024K
for bsz in 1 2
do
    for round in 1
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --context_len 1024000 \
            --task_name NIAH \
            --batch_size $bsz > different_lengths_logs/retroinfer_1024k_bsz${bsz}_${round}.log 2>&1
    done
done

for bsz in 4
do
    for round in 1
    do
        numactl --cpunodebind=0 --membind=0,1 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --context_len 1024000 \
            --task_name NIAH \
            --batch_size $bsz > different_lengths_logs/retroinfer_1024k_bsz${bsz}_${round}.log 2>&1
    done
done

unset CUDA_VISIBLE_DEVICES
