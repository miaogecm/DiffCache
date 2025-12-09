import time
import torch
from termcolor import colored
from tqdm import tqdm
from codetiming import Timer
import os
from flash_attn import flash_attn_with_kvcache


class BaseModel:
    def __init__(
        self, 
        model_name: str,
        max_length: int,
        dtype: torch.dtype,
        prefill_cache: bool = True
    ) -> None:
        """ Initializes the LLM.
        Args:
            model_name (str): The name of the model.
            max_length (int): The maximum length (prefill+decode) of sequences.
            dtype (torch.dtype): The data type for model computations.
        """

        self.model_name = model_name
        self.max_length = max_length
        self.dtype = dtype
        self.device_map = 'cuda:0'
        if prefill_cache:
            self.prefill_cache_dir = f"/tmp/diffcache_prefix/{model_name}"
            os.makedirs(self.prefill_cache_dir, exist_ok=True)
        else:
            self.prefill_cache_dir = None
        self.query_save_dir = None

        # TODO: only bf16 is supported for now
        assert(dtype == torch.bfloat16), "Only torch.bfloat16 is supported for now."

    def post_model_init(self):
        self.decode_pre_attn_graphs = [torch.cuda.CUDAGraph() for _ in range(self.num_layers)]
        self.decode_post_attn_graphs = [torch.cuda.CUDAGraph() for _ in range(self.num_layers)]

        self.kcache = [None for _ in range(self.num_layers)]
        self.vcache = [None for _ in range(self.num_layers)]

    def post_init_kv_cache(self):
        self.decode_hidden_states = torch.empty((self.batch_size, 1, self.hidden_size), dtype=self.dtype, device=self.device_map)
        self.decode_query_states = [None for _ in range(self.num_layers)]
        self.decode_key_states = [None for _ in range(self.num_layers)]
        self.decode_value_states = [None for _ in range(self.num_layers)]
        self.decode_position_ids = torch.zeros((self.batch_size, 1), dtype=torch.int32, device=self.device_map)
        self.decode_cache_seqlens = torch.zeros((self.batch_size,), dtype=torch.int32, device=self.device_map)

        for layer_idx in range(self.num_layers):
            self.build_layer_decode_graphs(layer_idx)

    def layer_prefill(self, layer_idx, hidden_states, chunk_size=8192):
        # print(f'Layer = {layer_idx}')
        bsz, seq_len, dim = hidden_states.shape
        layer = self.layers[layer_idx]
        
        # original hidden_states used as residual, clone a new one to process
        temp_hidden_states = hidden_states.clone()

        # chunk for lower memory comsumption
        for start_idx in range(0, seq_len, chunk_size // bsz):
            end_idx = min(seq_len, start_idx + chunk_size // bsz)
            temp_hidden_states[:, start_idx:end_idx, :] = self.layernorm(temp_hidden_states[:, start_idx:end_idx, :], 
                                                                         layer.input_layernorm_variance_epsilon, 
                                                                         layer.input_layernorm_weight)
        
        query_states, key_states, value_states = self.wqkv(temp_hidden_states, layer)

        del temp_hidden_states
        torch.cuda.empty_cache()
        query_states, key_states = self.position_embedd(query_states, key_states, 0)

        query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim)       # reshape [bs, seq_len, dim] => [bs, seq_len, head, head_dim]
        key_states = key_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        with Timer(f"prefill_update", logger=None):
            key_states, value_states = self.kv_cache.prefill_update_kv_cache(key_states, value_states, layer_idx)
            torch.cuda.empty_cache()

        torch.cuda.synchronize()
        with Timer(f"prefill_attn", logger=None):
            temp_attn_out = self.prefill_attention(query_states, key_states, value_states)
            torch.cuda.synchronize()

        del query_states, key_states, value_states
        torch.cuda.empty_cache()

        hidden_states += self.wo(temp_attn_out, layer, temp_attn_out.shape[0], seq_len, dim)
        del temp_attn_out
        torch.cuda.empty_cache()

        # post attention
        residual = hidden_states.clone()

        # chunk for lower memory comsumption
        for start_idx in range(0, seq_len, chunk_size // bsz):
            end_idx = min(seq_len, start_idx + chunk_size // bsz)
            hidden_states[:, start_idx:end_idx, :] = self.layernorm(hidden_states[:, start_idx:end_idx, :], 
                                                                    layer.post_attention_layernorm_variance_epsilon, 
                                                                    layer.post_attention_layernorm_weight)
            hidden_states[:, start_idx:end_idx, :] = self.mlp(hidden_states[:, start_idx:end_idx, :], layer)   
        
        hidden_states += residual

        del residual
        torch.cuda.empty_cache()
                                                                                                   
        return hidden_states
    
    def build_layer_decode_graphs(self, layer_idx):
        bsz, seq_len, dim = self.decode_hidden_states.shape
        layer = self.layers[layer_idx]

        # pre-attention graph
        with torch.cuda.graph(self.decode_pre_attn_graphs[layer_idx]):
            hidden_states = self.layernorm(self.decode_hidden_states, layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight)
            query_states, key_states, value_states = self.wqkv(hidden_states, layer)
            query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, self.decode_position_ids)
            self.kv_cache.decode_pre_query_operation(query_states)
        
        self.decode_query_states[layer_idx] = query_states.view(bsz, -1, self.num_heads, self.head_dim)
        self.decode_key_states[layer_idx] = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim)
        self.decode_value_states[layer_idx] = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim)

        # post-attention graph
        with torch.cuda.graph(self.decode_post_attn_graphs[layer_idx]):
            self.kv_cache.decode_post_query_operation(layer_idx)
            attn_out = flash_attn_with_kvcache(
                q=self.decode_query_states[layer_idx],
                k=self.decode_key_states[layer_idx],
                v=self.decode_value_states[layer_idx],
                cache_seqlens=self.decode_cache_seqlens,
                k_cache=self.kcache[layer_idx],
                v_cache=self.vcache[layer_idx],
                causal=True
            )
            hidden_states = self.wo(attn_out, layer, bsz, seq_len, dim)
            hidden_states = self.decode_hidden_states + hidden_states
            residual = hidden_states
            hidden_states = self.layernorm(hidden_states, layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight)
            hidden_states = self.mlp(hidden_states, layer)
            torch.add(hidden_states, residual, out=self.decode_hidden_states)

    def layer_decode(self, layer_idx):
        self.decode_position_ids.fill_(self.kv_cache.seq_len(layer_idx))
        self.decode_cache_seqlens.fill_(self.kv_cache.seq_len_gpu(layer_idx))
        self.decode_pre_attn_graphs[layer_idx].replay()
        self.kv_cache.decode_query_kvcache(layer_idx)
        self.decode_post_attn_graphs[layer_idx].replay()

    def do_prefill_forward(self, inputs_ids, prefill_cache_path):
        bsz, seq_len = inputs_ids.shape
        device = inputs_ids.device

        last_hidden_states = torch.empty((bsz, 1, self.hidden_size), dtype=self.dtype, device=device)
        hidden_states = self.word_embedding(inputs_ids)  # [bsz, seq_len, hidden_size]

        for ldx in tqdm(range(self.num_layers)):
            hidden_states = self.layer_prefill(ldx, hidden_states)
            torch.cuda.empty_cache()
        last_hidden_states[:, 0, :] = hidden_states[:, -1, :]
        
        last_hidden_states = self.layernorm(last_hidden_states.contiguous(), self.norm_variance_epsilon, self.norm_weight)
        logits = self.lm(last_hidden_states)

        if prefill_cache_path:
            torch.save(logits, os.path.join(prefill_cache_path, "logits.pt"))
            self.kv_cache.save_prefill_cache(prefill_cache_path)
        
        return logits
    
    # get prefill cache with matching input_ids
    def get_prefill_cache_path(self, input_ids):
        if self.prefill_cache_dir is None:
            return None
        for dir in os.listdir(self.prefill_cache_dir):
            dir_path = os.path.join(self.prefill_cache_dir, dir)
            if not os.path.isdir(dir_path) or not os.path.exists(os.path.join(dir_path, "input_ids.pt")):
                continue
            cached_input_ids = torch.load(os.path.join(dir_path, "input_ids.pt"))
            if torch.equal(cached_input_ids, input_ids):
                print(colored(f"Found matching prefill cache in {dir_path}\n", 'green'))
                return dir_path
        return None

    def create_prefill_cache_path(self, input_ids):
        if self.prefill_cache_dir is None:
            return None
        time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        dir_path = os.path.join(self.prefill_cache_dir, f"c{time_str}")
        os.makedirs(dir_path, exist_ok=False)
        return dir_path

    def prefill_forward(self, inputs_ids):
        prefill_cache_path = self.get_prefill_cache_path(inputs_ids)
        if prefill_cache_path is None:
            prefill_cache_path = self.create_prefill_cache_path(inputs_ids)
            result = self.do_prefill_forward(inputs_ids, prefill_cache_path)
            # save inputs_ids after everything is done
            torch.save(inputs_ids, os.path.join(prefill_cache_path, "input_ids.pt"))
            return result
        else:
            print(f"Loading prefill cache from {prefill_cache_path}")
            self.kv_cache.load_prefill_cache(inputs_ids, prefill_cache_path)
            logits = torch.load(os.path.join(prefill_cache_path, "logits.pt"))
            return logits

    def decode_forward(self, inputs_ids):
        self.decode_hidden_states.copy_(self.word_embedding(inputs_ids))

        for ldx in range(self.num_layers):
            self.layer_decode(ldx)
        
        hidden_states = self.layernorm(self.decode_hidden_states[:, -1:, :], self.norm_variance_epsilon, self.norm_weight)
        logits = self.lm(hidden_states)
        
        return logits

    def inference(self, inputs_ids):
        outputs_ids = []    # multi iteration, multi request
        output_ids = []     # single iteration, multi request
        
        print("Start prefilling ...")
        torch.cuda.synchronize()
        prefill_start = time.time()

        logits = self.prefill_forward(inputs_ids=inputs_ids)
        output_ids = logits.argmax(dim=-1)
        outputs_ids.append(output_ids)

        torch.cuda.synchronize()
        prefill_end = time.time()
        print(colored(f"Prefilling latency: {round((prefill_end - prefill_start), 4)} s\n", 'green'))

        for layer_idx in range(self.num_layers):
            self.kcache[layer_idx], self.vcache[layer_idx] = self.kv_cache.get_kvcache(layer_idx)

        self.post_init_kv_cache()

        batch_size = inputs_ids.shape[0]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=inputs_ids.device)

        print("Start decoding ...")
        decode_start = time.time()

        for i in tqdm(range(self.max_new_length-1)):
            logits = self.decode_forward(inputs_ids=output_ids)
            output_ids = logits.argmax(dim=-1)

            # check finished sequences
            output_ids = torch.where(finished, torch.full_like(output_ids, self.config.eos_token_id), output_ids)
            finished |= output_ids[:, 0].eq(self.config.eos_token_id)
            if finished.all():
                break

            outputs_ids.append(output_ids)

        decode_end = time.time()
        print(colored(
            f"Decoding latency: {round((decode_end - decode_start) * 1000 / (self.max_new_length - 1), 2)} ms/step, "
            f"Throughput: {round(self.batch_size * (self.max_new_length - 1) / (decode_end - decode_start), 2)} tokens/s\n",
            'green'
        ))

        if self.query_save_dir is not None:
            print("Saving queries...")
            os.makedirs(self.query_save_dir, exist_ok=True)
            self.kv_cache.save_queries(self.query_save_dir)
        
        outputs_ids = torch.cat(outputs_ids, dim=-1).tolist()
        
        return outputs_ids

    def generate(self, inputs_ids, attention_masks, max_new_length, attn_config=None):
        """ LLM Inference.
        Args:
            input_ids (torch.tensor): The input of LLM.
            attention_masks (torch.tensor): The attention masks of LLM.
            max_new_length (int): The maximum length of generated sequences.
        """

        bs, input_length = inputs_ids.shape
        assert input_length + max_new_length <= self.max_length, \
        f"Error: input_length({input_length}) + max_new_length({max_new_length}) exceeds max_length({self.max_length})"

        self.batch_size = bs
        self.input_length = input_length
        self.max_new_length = max_new_length

        valid_start = attention_masks.shape[1] - torch.sum(attention_masks, dim=-1).detach().cpu().numpy()
        del attention_masks
        torch.cuda.empty_cache()

        print("Initializing KVCache ...\n")
        self.init_kv_cache(input_length, valid_start, attn_config)

        outputs = self.inference(inputs_ids)

        return outputs
