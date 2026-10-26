# Print raw model answers next to COIN labels, through the same dataset -> collator -> generate_after_embed path
# as evaluate.py, to check that low accuracy is the model and not the scoring.
# Usage: python scripts/coin/inspect_predictions.py <checkpoint> [<checkpoint> ...]   (run from the repo root, 1 GPU)
import sys, random, torch
from dataclasses import asdict
from models import build_model_and_tokenizer
from models.arguments_live import get_args_class
from data import build_eval_dataset_dict, get_data_collator

NUM_SAMPLES = {'coin_step_test': 8, 'coin_task_test': 6, 'coin_procedure_test': 4, 'coin_taskprocedure_test': 4}

def main(checkpoints):
    for checkpoint in checkpoints:
        args = get_args_class('live1+')(output_dir='outputs/debug', attn_implementation='sdpa', resume_from_checkpoint=checkpoint,
                                        eval_datasets=list(NUM_SAMPLES))
        model, tokenizer = build_model_and_tokenizer(is_training=False, **asdict(args))
        model.to('cuda', torch.bfloat16)
        datasets = build_eval_dataset_dict(tokenizer=tokenizer, model_config=model.config, **asdict(args))
        collator = get_data_collator(tokenizer=tokenizer, model_config=model.config, **asdict(args))
        print(f'\n################ {checkpoint}')
        for name, dataset in datasets.items():
            idxs = random.Random(0).sample(range(len(dataset)), NUM_SAMPLES[name]) # same questions for every checkpoint
            correct = 0
            print(f'\n=== {name} ({len(dataset)} questions, showing {len(idxs)})')
            for idx in idxs:
                inputs = collator([dataset[idx]])
                inputs.pop('sample_idxs')
                kwargs = dict(inputs.pop('evaluation_kwargs'))
                evaluator = kwargs.pop('evaluator')
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
                with torch.no_grad():
                    output_ids = getattr(model, evaluator)(**inputs, **kwargs, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                output_ids[output_ids < 0] = tokenizer.bos_token_id
                prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                label = dataset.labels[idx]
                if isinstance(label, str): # step / next / task: one answer, scored like COINBenchmark.compute_metrics
                    pred = prediction.lower().rstrip('.')
                    matched = dataset.fuzzy_match(pred, dataset.categories)
                    ok = pred == label or matched == label
                    correct += ok
                    print(f'[{"OK " if ok else "BAD"}] answer: {prediction!r}\n      matched: {matched!r} | label: {label!r}')
                else: # procedure: per-step, scored like COINProcedure.compute_metrics
                    steps = [s.split('. ')[-1] for s in prediction.split('\n')]
                    marks = ['OK' if (p == l or dataset.fuzzy_match(p, dataset.categories) == l) else 'BAD' for p, l in zip(steps, label)]
                    correct += marks.count('OK') / len(label)
                    print(f'[{"/".join(marks)}] answer: {prediction!r}\n      labels: {list(label)}')
            print(f'--- {correct:.1f}/{len(idxs)} correct on this sample')
        del model
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main(sys.argv[1:])
