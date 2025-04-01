import torch
import numpy as np
from torch.utils.data import DataLoader
from data import  collate_fn, concat_with_valid_tokens_first, EmiliaIterableDataset
from models import Model
import utils
from utils import dist
from tqdm import tqdm

utils.set_seed_all(42)
dist.init()
torch.cuda.set_device(dist.local_rank())


# --- Config ---
CKPT_PATH='yycc/csm-1b-chinese'
NUM_GRAD_ACCUM=10
NUM_AUDIO_DIM=32
LR=1e-5
device = 'cuda'
dtype = torch.bfloat16 # Target dtype for autocast operations


# --- Mixed Precision Setup ---
from torch.amp import GradScaler, autocast # Import autocast and GradScaler
scaler = GradScaler()

model = Model.from_pretrained(CKPT_PATH)
model.to(device=device) # Keep model parameters in float32, autocast handles internal ops
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# data
dataloader = DataLoader(
    EmiliaIterableDataset(),
    batch_size=8,
    collate_fn=collate_fn,
    num_workers=8*2,
)

# Learning rate scheduler with warmup
warmup_steps = 500
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: min(float(step + 1) / warmup_steps, 1.0)
)

losses = []

mimi = utils.load_audio_tokenizer(device=device) # audio tokenizer:
model.train() # Ensure model is in training mode
optimizer.zero_grad() # Initialize gradients to zero before starting

for epoch in range(100):
    for idx,batch in tqdm(enumerate(dataloader,1)):
        # process input and tokenize audio
        audio_tokens = mimi.encode(batch['audio'][:,None].float().cuda())
        invalid_audio_mask = (batch['valid_audio_mask'].cuda()==0)[:,None].repeat(1,NUM_AUDIO_DIM,1)
        audio_tokens[invalid_audio_mask]=-1
        audio_tokens = torch.cat([audio_tokens, audio_tokens[:,0:1]*0-1],dim=1).permute(0,2,1) # (batch_size, seq_len, audio_num_codebooks+1), last token is -1 (text)
        # NOTE: a bit tricky, to be refactored, basically we want to concat text and audio tokens, they are both padded, we want to keep the valid tokens first, move all padding to the end
        all_tokens = concat_with_valid_tokens_first(batch['text_tokens'].cuda(), audio_tokens, batch['valid_text_mask'].bool().cuda(), batch['valid_audio_mask'].bool().cuda())
        all_mask = concat_with_valid_tokens_first(batch['text_mask'].cuda().float(), (audio_tokens.cuda()>-0.5).float(), batch['valid_text_mask'].bool().cuda(), batch['valid_audio_mask'].bool().cuda())>0

        all_mask = all_mask.bool()
        all_mask.flatten(0,1)[all_mask.flatten(0,1).sum(-1)>30,-1]=False

        # --- Mixed Precision Forward &Backward Pass ---
        with autocast('cuda', dtype=dtype): # Enable autocast context
            res = model(all_tokens, all_mask)
            loss = res['loss']
            # Scale loss for gradient accumulation
            loss = loss / NUM_GRAD_ACCUM
        scaler.scale(loss).backward()

        current_loss = loss.item() * NUM_GRAD_ACCUM # Store un-accumulated loss for logging
        losses.append(current_loss)

        if (idx)%NUM_GRAD_ACCUM==0:
            utils.dist_sync_grad(model) # Sync gradients across devices (BEFORE unscaling)
            scaler.step(optimizer)

            scaler.update()
            scheduler.step()
            optimizer.zero_grad()


        if idx%100==0 and dist.rank()==0:
            print(f"Epoch: {epoch}, Step: {idx}, Avg Loss (last 100): {np.mean(losses[-100:]):.4f}") # Use current_loss for logging
        if idx %5000==0 and dist.rank()==0:
            save_path = f"nmodel_{idx}"
            print(f"Saving model to {save_path}")
            # os.makedirs(f"model_{idx}", exist_ok=True) # Not needed for save_pretrained
            model.save_pretrained(save_path)
