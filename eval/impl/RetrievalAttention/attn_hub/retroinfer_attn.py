from flash_attn import flash_attn_with_kvcache


def retroinfer_prefill_attn(query_states, key_states, value_states, causal):

    attn_out = flash_attn_with_kvcache(
        q=query_states, 
        k_cache=key_states, 
        v_cache=value_states,
        causal=causal
    )
    
    return attn_out



def retroinfer_decode_attn(query_states, key_states, value_states, layer_idx, retroinfer_cache):
    
    attn_out = retroinfer_cache.compute(
        query_states.contiguous(), layer_idx
    )
    
    return attn_out
