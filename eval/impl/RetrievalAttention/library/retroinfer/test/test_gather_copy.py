import torch
import numpy as np
import time
import math
from retroinfer_kernels import gather_copy_and_concat, gather_copy_and_scatter, gather_copy_vectors
import random

DTYPE = torch.bfloat16


# generate two random indices with shape (rows, cols)
def gen_two_indices(rows, cols, max_range1, max_range2, unit_size):
    src_indices1 = np.random.randint(-1, 100, size=(rows, cols), dtype=np.int32)
    src_copy_size1 = np.random.randint(1, unit_size+1, size=(rows, cols), dtype=np.int32)
    dst_indices1 = np.random.randint(0, 100, size=(rows, cols), dtype=np.int32)
    copy_chunks1 = np.random.randint(0, 10, size=(rows,), dtype=np.int32)

    src_indices2 = np.random.randint(-1, 100, size=(rows, cols), dtype=np.int32)
    src_copy_size2 = np.random.randint(1, unit_size+1, size=(rows, cols), dtype=np.int32)
    dst_indices2 = np.random.randint(0, 100, size=(rows, cols), dtype=np.int32)
    copy_chunks2 = np.random.randint(0, 10, size=(rows,), dtype=np.int32)

    for i in range(rows):
        num = np.random.randint(int(0.8*cols), cols)
        if i == 1:
            num1 = 0
            num2 = num
        elif i == 5:
            num1 = num
            num2 = 0
        else:
            num1 = np.random.randint(0, num)
            num2 = num - num1

        src_indices1[i, :num1] = np.random.choice(max_range1, num1, replace=False)
        cumsum = 0
        for j in range(num1):
            copy_size = np.random.randint(0, unit_size+1)   # [0, unit_size]
            src_copy_size1[i, j] = copy_size
            dst_indices1[i, j] = cumsum
            cumsum += copy_size
        # 设定边界拷贝
        if num1 > 0:
            x = np.random.randint(0, num1)
            src_indices1[i, x] = max_range1+unit_size-src_copy_size1[i, x]
        copy_chunks1[i] = num1

        src_indices2[i, :num2] = np.random.choice(max_range2, num2, replace=False)
        cumsum = 0
        for j in range(num2):
            copy_size = np.random.randint(0, unit_size+1)   # [0, unit_size]
            src_copy_size2[i, j] = copy_size
            dst_indices2[i, j] = cumsum
            cumsum += copy_size
        copy_chunks2[i] = num2

    src_indices1 = torch.from_numpy(src_indices1).pin_memory()
    src_copy_size1 = torch.from_numpy(src_copy_size1).pin_memory()
    dst_indices1 = torch.from_numpy(dst_indices1).pin_memory()
    copy_chunks1 = torch.from_numpy(copy_chunks1).pin_memory()

    src_indices2 = torch.from_numpy(src_indices2).pin_memory()
    src_copy_size2 = torch.from_numpy(src_copy_size2).pin_memory()
    dst_indices2 = torch.from_numpy(dst_indices2).pin_memory()
    copy_chunks2 = torch.from_numpy(copy_chunks2).pin_memory()
    return src_indices1, src_copy_size1, dst_indices1, copy_chunks1, src_indices2, src_copy_size2, dst_indices2, copy_chunks2

def test_concat_gather_copy():
    groups = 8
    src_vector_num1 = 1769
    src_vector_num2 = 12397
    src_unit_num3 = 1000
    buffer_unit_num = 400
    index_length = 400
    unit_size = 8
    dim = 128
    copy_vector_num = 1602
    buffer_vector_num = buffer_unit_num * unit_size + src_vector_num1

    key_src1 = torch.randn((groups, src_vector_num1, dim), device='cuda', dtype=DTYPE).contiguous()
    key_src2 = torch.randn((groups, src_vector_num2, dim), pin_memory=True, dtype=DTYPE).contiguous()
    key_src3 = torch.randn((groups, src_unit_num3, unit_size, dim), device='cuda', dtype=DTYPE).contiguous()
    key_dst1 = torch.randn((groups, buffer_vector_num, dim), device='cuda', dtype=DTYPE).contiguous()
    key_dst2 = key_dst1.clone()
    
    value_src1 = torch.randn((groups, src_vector_num1, dim), device='cuda', dtype=DTYPE).contiguous()
    value_src2 = torch.randn((groups, src_vector_num2, dim), pin_memory=True, dtype=DTYPE).contiguous()
    value_src3 = torch.randn((groups, src_unit_num3, unit_size, dim), device='cuda', dtype=DTYPE).contiguous()
    value_dst1 = torch.randn((groups, buffer_vector_num, dim), device='cuda', dtype=DTYPE).contiguous()
    value_dst2 = value_dst1.clone()

    valid_lengths = torch.empty((groups,), dtype=torch.int32, pin_memory=True)

    src_indices1, src_copy_size1, dst_indices1, copy_chunks1, src_indices2, src_copy_size2, dst_indices2, copy_chunks2 = gen_two_indices(groups, index_length, src_vector_num2-unit_size, src_unit_num3, unit_size)
    torch.cuda.synchronize()

    t1 = time.time()
    gather_copy_and_concat(key_src1, key_src2, key_src3, key_dst1,
                           value_src1, value_src2, value_src3, value_dst1,
                           src_indices1, src_copy_size1, dst_indices1, copy_chunks1,
                           src_indices2, src_copy_size2, dst_indices2, copy_chunks2,
                           valid_lengths, groups, src_vector_num1, src_vector_num2, src_unit_num3, 
                           buffer_vector_num, index_length, copy_vector_num)

    torch.cuda.synchronize()
    print("cuda time: ", time.time()-t1)

    print("valid_lengths: ", valid_lengths)

    for i in range(groups):
        print(f"group{i}, {copy_chunks1[i]}, {copy_chunks2[i]}")

        key_dst2[i, :copy_vector_num, :] = key_src1[i, :copy_vector_num, :]
        value_dst2[i, :copy_vector_num, :] = value_src1[i, :copy_vector_num, :]
        copy_num = copy_vector_num

        for j in range(copy_chunks1[i]):
            key_dst2[i, copy_num:copy_num+src_copy_size1[i, j], :] = key_src2[i, src_indices1[i, j]:src_indices1[i, j]+src_copy_size1[i, j], :]
            value_dst2[i, copy_num:copy_num+src_copy_size1[i, j], :] = value_src2[i, src_indices1[i, j]:src_indices1[i, j]+src_copy_size1[i, j], :]
            copy_num += src_copy_size1[i, j]
        
        for j in range(copy_chunks2[i]):
            key_dst2[i, copy_num:copy_num+src_copy_size2[i, j], :] = key_src3[i, src_indices2[i, j], :src_copy_size2[i, j], :]
            value_dst2[i, copy_num:copy_num+src_copy_size2[i, j], :] = value_src3[i, src_indices2[i, j], :src_copy_size2[i, j], :]
            copy_num += src_copy_size2[i, j]
        
        assert copy_num == valid_lengths[i], f"{i, copy_num, valid_lengths[i]}"
    
    assert (key_dst1 == key_dst2).all()
    assert (value_dst1 == value_dst2).all()



def gen_indices(rows, cols, max_range1, max_range2, unit_size):
    src_indices = np.random.randint(-1, 100, size=(rows, cols), dtype=np.int32)
    src_copy_size = np.random.randint(1, unit_size+1, size=(rows, cols), dtype=np.int32)
    dst_indices = np.random.randint(0, 100, size=(rows, cols), dtype=np.int32)
    copy_chunks = np.random.randint(0, 10, size=(rows,), dtype=np.int32)

    for i in range(rows):
        if i == 1:
            copy_chunks[i] = 0
            continue
        
        num = np.random.randint(int(0.2*cols), int(0.8*cols))
        
        src_indices[i, :num] = np.random.choice(max_range1, num, replace=False)
        dst_indices[i, :num] = np.random.choice(max_range2, num, replace=False)
        for j in range(num):
            copy_size = np.random.randint(0, unit_size+1)   # [0, unit_size]
            if src_indices[i, j] + copy_size > max_range1:  # overflow
                copy_size = max_range1 - src_indices[i, j]
            src_copy_size[i, j] = copy_size
        copy_chunks[i] = num

    src_indices = torch.from_numpy(src_indices).pin_memory()
    src_copy_size = torch.from_numpy(src_copy_size).pin_memory()
    dst_indices = torch.from_numpy(dst_indices).pin_memory()
    copy_chunks = torch.from_numpy(copy_chunks).pin_memory()

    return src_indices, src_copy_size, dst_indices, copy_chunks

def test_gather_copy_scatter():
    groups = 8
    src_unit_num = 400
    dst_unit_num = 1000
    index_length = 400
    unit_size = 8
    dim = 128
    copy_start = 97

    key_src = torch.randn((groups, src_unit_num*unit_size, dim), device='cuda', dtype=DTYPE).contiguous()
    key_dst1 = torch.randn((groups, dst_unit_num, unit_size, dim), device='cuda', dtype=DTYPE).contiguous()
    key_dst2 = key_dst1.clone()
    
    value_src = torch.randn((groups, src_unit_num*unit_size, dim), device='cuda', dtype=DTYPE).contiguous()
    value_dst1 = torch.randn((groups, dst_unit_num, unit_size, dim), device='cuda', dtype=DTYPE).contiguous()
    value_dst2 = value_dst1.clone()

    src_indices, src_copy_size, dst_indices, copy_chunks = gen_indices(groups, index_length, src_unit_num*unit_size-copy_start, dst_unit_num, unit_size)
    torch.cuda.synchronize()

    t1 = time.time()
    gather_copy_and_scatter(key_src, key_dst1, value_src, value_dst1, 
                            src_indices, src_copy_size, dst_indices, copy_chunks, 
                            groups, src_unit_num*unit_size, dst_unit_num, index_length, copy_start)
    torch.cuda.synchronize()
    print("cuda time: ", time.time()-t1)

    for i in range(groups):
        print(f"group{i}, {copy_chunks[i]}")
        for j in range(copy_chunks[i]):
            key_dst2[i, dst_indices[i, j], :src_copy_size[i, j], :] = key_src[i, copy_start+src_indices[i, j]:copy_start+src_indices[i, j]+src_copy_size[i, j], :]
            value_dst2[i, dst_indices[i, j], :src_copy_size[i, j], :] = value_src[i, copy_start+src_indices[i, j]:copy_start+src_indices[i, j]+src_copy_size[i, j], :]
    
    assert (key_dst1 == key_dst2).all()
    assert (value_dst1 == value_dst2).all()



def test_gather_copy_vectors():
    groups = 8
    src_vector_num = 8192
    dim = 128
    nprobe = 150
    index_size = 2048
    copy_vector_num = index_size - nprobe
    buffer_size = copy_vector_num

    key_src = torch.randn((groups, src_vector_num, dim), device='cuda', dtype=DTYPE).contiguous()
    key_dst1 = torch.randn((groups, buffer_size, dim), device='cuda', dtype=DTYPE).contiguous()
    key_dst2 = key_dst1.clone()
    
    value_src = torch.randn((groups, src_vector_num, dim), device='cuda', dtype=DTYPE).contiguous()
    value_dst1 = torch.randn((groups, buffer_size, dim), device='cuda', dtype=DTYPE).contiguous()
    value_dst2 = value_dst1.clone()

    src_metadata = torch.randn(size=(groups, src_vector_num), dtype=DTYPE, device='cuda').contiguous()
    dst_metadata1 = torch.empty((groups, buffer_size), dtype=DTYPE, device='cuda').contiguous()
    dst_metadata2 = dst_metadata1.clone()

    indices = torch.empty((groups, index_size), dtype=torch.int64, device='cuda')
    for i in range(groups):
        indices[i, :] = torch.randperm(src_vector_num)[:index_size].to(torch.int64).to("cuda")
    
    torch.cuda.synchronize()
    start = time.time()
    gather_copy_vectors(key_src, key_dst1, value_src, value_dst1, src_metadata, dst_metadata1, 
                        indices, groups, src_vector_num, buffer_size, index_size, nprobe, copy_vector_num)
    torch.cuda.synchronize()
    print("cuda time: ", time.time()-start)
    
    copy_indices = indices[:, nprobe:nprobe+copy_vector_num]
    for i in range(groups):
        key_dst2[i, :copy_vector_num, :] = key_src[i, copy_indices[i, :], :]
        value_dst2[i, :copy_vector_num, :] = value_src[i, copy_indices[i, :], :]
        dst_metadata2[i, :copy_vector_num] = src_metadata[i, copy_indices[i, :]]

    assert (key_dst1 == key_dst2).all()
    assert (value_dst1 == value_dst2).all()
    assert (dst_metadata1 == dst_metadata2).all()



if __name__ == "__main__":
    for i in range(10):
        test_concat_gather_copy()
        test_gather_copy_scatter()
        test_gather_copy_vectors()
        print("pass")
    