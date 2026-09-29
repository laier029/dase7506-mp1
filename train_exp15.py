"""Longer training recipe that saves the best validation checkpoint."""
import argparse
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score


def main():
    total_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=ROOT/'configs/baseline.json')
    p.add_argument('--run-dir', type=Path, default=ROOT/'runs/baseline-s17')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto','fp32','bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=2400)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--eval-every', type=int, default=300,
                   help='Validation and best-checkpoint interval.')
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.eval_every < 1:
        p.error('Batch size, step count and eval interval must be positive.')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Run directory already contains results. Use a new --run-dir.')
    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_data()
    config = json.loads(args.config.read_text())
    model, implementation_sha = make_model(args.implementation, config, device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.1)
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter()-prepared
    started = time.perf_counter()
    history = []
    validation_history = []
    intermediate_validation_seconds = 0.
    best_bpb = float('inf')
    best_step = None
    best_validation = None
    checkpoint = args.run_dir/'checkpoint.pt'

    def save_checkpoint(path, step):
        state = {name: value.detach().cpu().clone()
                 for name, value in model.state_dict().items()}
        torch.save({'protocol':PROTOCOL,'implementation':args.implementation,
                    'config':config,'model':state,'seed':args.seed,
                    'train_tokens':step*args.batch_size*256},path)

    for step in range(args.steps):
        starts = torch.randint(len(tokens)-257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:,None]+torch.arange(257,device=device)]
        learning_rate = .001 * min(1.,(step+1)/100) * (.1+.9*.5*(1+math.cos(math.pi*step/args.steps)))
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            loss = F.cross_entropy(model(batch[:,:-1]).flatten(0,1).float(),batch[:,1:].flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if (step+1)%100 == 0 or step+1 == args.steps:
            row = {'step':step+1,'loss':loss.item(),'seconds':time.perf_counter()-started-intermediate_validation_seconds}
            history.append(row)
            print(json.dumps(row),flush=True)
        if (step+1)%args.eval_every == 0:
            intermediate = score(model,*data['validation'],device,'fp32')
            intermediate.pop('window_nll_nats')
            intermediate_validation_seconds += intermediate['seconds']
            validation_history.append({'step':step+1,**intermediate})
            print(json.dumps({'validation':validation_history[-1]}),flush=True)
            if intermediate['bpb'] < best_bpb:
                best_bpb = intermediate['bpb']
                best_step = step+1
                best_validation = dict(intermediate)
                save_checkpoint(checkpoint, best_step)
                print(json.dumps({'new_best_step':best_step,
                                  'new_best_validation_bpb':best_bpb}),flush=True)

    if args.steps % args.eval_every:
        intermediate = score(model,*data['validation'],device,'fp32')
        intermediate.pop('window_nll_nats')
        intermediate_validation_seconds += intermediate['seconds']
        validation_history.append({'step':args.steps,**intermediate})
        if intermediate['bpb'] < best_bpb:
            best_bpb = intermediate['bpb']
            best_step = args.steps
            best_validation = dict(intermediate)
            save_checkpoint(checkpoint, best_step)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter()-started-intermediate_validation_seconds
    last_checkpoint = args.run_dir/'last_checkpoint.pt'
    save_checkpoint(last_checkpoint, args.steps)
    validation = dict(best_validation)
    result = {'protocol':PROTOCOL,'implementation':args.implementation,'config':config,'seed':args.seed,
              'parameters':sum(p.numel() for p in model.parameters()),'precision':precision,
              'train_tokens':best_step*args.batch_size*256,
              'search_train_tokens':args.steps*args.batch_size*256,
              'best_step':best_step,'preparation_seconds':preparation_seconds,
              'train_seconds':train_seconds,'validation':validation,'history':history,
              'validation_history':validation_history,
              'intermediate_validation_seconds':intermediate_validation_seconds,
              'process_seconds':time.perf_counter()-total_started,
              'torch_version':str(torch.__version__),'threads':args.threads,
              'checkpoint_sha256':sha(checkpoint),
              'last_checkpoint_sha256':sha(last_checkpoint),
              'implementation_sha256':implementation_sha,
              **device_metrics(device)}
    (args.run_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result|{'history':[]},indent=2),flush=True)


if __name__ == '__main__':
    main()
