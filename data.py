import torch
import torchaudio
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from typing import Tuple
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing
from utils import dist

def load_llama3_tokenizer():
    """
    https://github.com/huggingface/transformers/issues/22794#issuecomment-2092623992
    """
    tokenizer_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
    )

    return tokenizer

def _tokenize_text_segment(tokenizer, text: str, speaker: int = 0, num_codebook=32) -> Tuple[torch.Tensor, torch.Tensor]:
    frame_tokens = []
    frame_masks = []

    text_tokens =tokenizer.encode(f"[{speaker}]{text}")
    text_frame = torch.zeros(len(text_tokens), num_codebook+1).long()
    text_frame_mask = torch.zeros(len(text_tokens),num_codebook+1).bool()
    text_frame[:, -1] = torch.tensor(text_tokens)
    text_frame_mask[:, -1] = True

    frame_tokens.append(text_frame)
    frame_masks.append(text_frame_mask)

    return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0).bool()


from torch.utils.data import IterableDataset

class EmiliaIterableDataset(IterableDataset):
    def __init__(self, path="Emilia/ZH/*.tar", split="zh", tokenizer=None, sample_rate=24000):
        self.tar_files = [f"Emilia/ZH/ZH-B{i:06d}.tar" for i in range(920)][dist.rank()::dist.size()]
        self._text_tokenizer = load_llama3_tokenizer()
        self.sample_rate = sample_rate
        self._dataset = load_dataset("amphion/Emilia-Dataset", 
                                   data_files={split: self.tar_files}, 
                                   split=split, 
                                   streaming=True)
       
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        self.dataset = self._dataset.shard(
                                           num_shards=num_workers, index=worker_id
                                           )
        print(worker_id, num_workers, dist.size(), dist.rank())
        for item in self.dataset:
            # Extract audio and text
            audio = item["mp3"]["array"]
            sample_rate = item["mp3"]["sampling_rate"]
            text = item["json"]["text"]
            
            # Resample audio if needed
            if sample_rate != self.sample_rate:
                audio = torch.tensor(audio)
                audio = torchaudio.functional.resample(audio, 
                                                      orig_freq=sample_rate, 
                                                      new_freq=self.sample_rate)
            else:
                audio = torch.tensor(audio)
            # trunk if needed
            audio = audio[:24000*10]
            
            # Tokenize text
            text_tokens, text_mask = _tokenize_text_segment(self._text_tokenizer, text)
            
            yield dict(
                text_tokens=text_tokens,
                text_mask=text_mask,
                text=text,
                audio=audio,
            )

# Updated collate function that works with our transformed data structure
def collate_fn(batch):
    # Extract individual elements
    text_tokens = [item['text_tokens'] for item in batch]
    text_mask = [item['text_mask'] for item in batch]
    audio = [item['audio'] for item in batch]
    text = [item['text'] for item in batch]
    
    # Get max lengths
    max_text_length = max([t.shape[0] for t in text_tokens])
    
    # Initialize tensors with padding
    batch_size = len(batch)
    text_tokens_dim = text_tokens[0].shape[1]  # Should be 33 based on comments
    
    # Prepare batched tensors for text
    batch_text_tokens = torch.zeros(batch_size, max_text_length, text_tokens_dim, dtype=text_tokens[0].dtype)
    batch_text_mask = torch.zeros(batch_size, max_text_length, text_tokens_dim, dtype=text_mask[0].dtype)
    batch_valid_text_mask = torch.zeros(batch_size, max_text_length)
    
    # Fill text data
    for i, (tokens, mask) in enumerate(zip(text_tokens, text_mask)):
        text_length = tokens.shape[0]
        batch_text_tokens[i, :text_length] = tokens
        batch_text_mask[i, :text_length] = mask
        batch_valid_text_mask[i, :text_length] = 1
    
    # Pad audio sequences
    batch_audio = pad_sequence(audio, batch_first=True)
    
    # Create audio mask to indicate real vs padded values
    max_audio_length = batch_audio.shape[1]
    batch_audio_valid_mask = torch.zeros(batch_size, max_audio_length)
    for i, a in enumerate(audio):
        batch_audio_valid_mask[i, :len(a)] = 1
    
    return {
        'text_tokens': batch_text_tokens,
        'text_mask': batch_text_mask,
        'valid_text_mask': batch_valid_text_mask,
        'text': text,
        'audio': batch_audio.float(),
        'valid_audio_mask': batch_audio_valid_mask[:,::1920],
    }


def concat_with_valid_tokens_first(A, B, A_mask, B_mask):
    """A is text, B is audio, both are padded tensors with shape [batch_size, N, c] and [batch_size, M, c] respectively.
    A_mask and B_mask are boolean tensors of shape [batch_size, N] and [batch_size, M] respectively.
    """
    # NOTE: Invalid tokens are set to -1
    assert A.dtype == B.dtype, "A and B must have the same dtype"
    # assert A.dtype==torch.float32, "A must be float32"

    batch_size, N, c = A.shape
    _, M, _ = B.shape
    
    # Create output tensor filled with padding value (-1), boolTensor will fail
    # NOTE: Filled with -1, if dtype is bool, it will be wrong
    assert A.dtype!=torch.bool, "bool tensor will fail"
    output = torch.full((batch_size, N+M, c), -1, dtype=A.dtype, device=A.device)
    
    # Get valid token counts for each batch
    valid_A_count = A_mask.sum(dim=1)  # [batch_size]
    
    # First, get all batch and sequence indices
    batch_indices = torch.arange(batch_size, device=A.device)
    
    # Create a range tensor that will be used to gather valid tokens in order
    # This is a tensor of shape [batch_size, max(N,M)] containing sequential indices
    seq_range = torch.arange(max(N, M), device=A.device).unsqueeze(0).expand(batch_size, -1)
    
    # For each batch, find indices of valid elements in A and B
    A_valid_indices = A_mask.nonzero(as_tuple=True)  # Returns (batch_indices, seq_indices)
    B_valid_indices = B_mask.nonzero(as_tuple=True)
    
    # Get values at valid positions
    A_values = A[A_valid_indices]  # Shape: [num_valid_A, c]
    B_values = B[B_valid_indices]  # Shape: [num_valid_B, c]
    
    # Create position mapping tensors
    A_positions = torch.zeros_like(A_mask, dtype=torch.long)
    B_positions = torch.zeros_like(B_mask, dtype=torch.long)
    
    # Use scatter to create position mappings
    # For each valid position in batch b, assign its sequential index
    A_positions.scatter_(1, seq_range[:,:N].masked_fill(~A_mask, 0), 
                        seq_range[:,:N].masked_fill(~A_mask, 0))
    
    # For B, position starts after all valid A tokens in same batch
    B_positions.scatter_(1, seq_range[:,:M].masked_fill(~B_mask, 0), 
                        seq_range[:,:M].masked_fill(~B_mask, 0) + valid_A_count.unsqueeze(1))
    
    # Get destination positions for all valid tokens
    A_dest_positions = A_positions[A_valid_indices]
    B_dest_positions = B_positions[B_valid_indices]
    
    # Scatter valid values to output tensor
    output[A_valid_indices[0], A_dest_positions] = A_values
    output[B_valid_indices[0], B_dest_positions] = B_values
    
    return output
if __name__=='__main__':
    dataset = self = EmiliaIterableDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=2,  # Adjust batch size as needed
        shuffle=False,   # Shuffle during training
        collate_fn=collate_fn,
        num_workers=4,
    )
    
    for batch in dataloader:
        # apply mask
        break
    

