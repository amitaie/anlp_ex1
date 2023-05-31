import argparse
import transformers
from datasets import load_dataset, DatasetDict
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    EvalPrediction
from evaluate import load
#import wandb
import os
import numpy as np


def load_and_slice(n_train=-1, n_val=-1, n_test=-1):
    train = f'train[:{n_train}]' if n_train != -1 else 'train[:]'
    val = f'validation[:{n_val}]' if n_val != -1 else 'validation[:]'
    test = f'test[:{n_test}]' if n_test != -1 else 'test[:]'
    ds_train, ds_val, ds_test = load_dataset('sst2', split=[train, val, test])
    return DatasetDict({'train': ds_train, 'validation': ds_val, 'test': ds_test})


def compute_metrics(p: EvalPrediction):
    metric = load("accuracy")
    preds, gt = p
    b_preds = np.argmax(preds, axis=1)
    result = metric.compute(predictions=b_preds, references=gt)
    return result


def select_best_model_data(training_data, stats):
    means = {k: v['mean'] for k, v in stats.items()}
    best_model_name = max(means, key=means.get)

    prev_acc = 0.0
    best_model_data = None
    for data in training_data:
        if data['model_name'] == best_model_name and data['eval_accuracy'] > prev_acc:
            best_model_data = data
            prev_acc = data['eval_accuracy']

    return best_model_data


def predict_and_save(model_name, best_model_trainer, test_dataset):
    best_model_trainer.model.eval()
    best_model_trainer.args.per_device_eval_batch_size = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokened_dataset = test_dataset.map(lambda examples: tokenizer(examples["sentence"], truncation=True))

    predictions = best_model_trainer.predict(tokened_dataset.remove_columns(['sentence', 'idx', 'label']))
    predict_time = predictions.metrics['test_runtime']
    predictions = np.argmax(predictions.predictions, axis=1)

    with open('predictions.txt', 'w') as f:
        for sample, pred in zip(tokened_dataset, predictions):
            f.write(f"{sample['sentence']}###{pred}\n")

    return predict_time


def write_stats(stats, predict_time, train_time):
    with open("res.txt", 'w') as f:
        for model_name, stat in stats.items():
            f.write(f"{model_name},{round(stat['mean'], 3)} +- {round(stat['std'], 3)}\n")
        f.write("----\n")
        f.write(f"train time,{train_time}\n")
        f.write(f"predict time,{predict_time}\n")


def main(args):
    output_dir = '/content/gdrive/MyDrive/anlp_ex1'
    ALL_MODELS = ['bert-base-uncased', 'roberta-base', 'google/electra-base-generator']

    sliced_datasets = load_and_slice(args.num_train_samp, args.num_val_samp, args.num_test_samp)

    training_data = []
    stats = dict.fromkeys(ALL_MODELS)
    train_time = 0.0

    for model_name in ALL_MODELS:
        model_res = []
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokened_datasets = sliced_datasets.map(lambda examples: tokenizer(examples["sentence"], truncation=True),
                                               batched=True)
        config = AutoConfig.from_pretrained(model_name)

        for seed in range(args.num_seeds):
            transformers.set_seed(seed)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
            exp_name = f"model_{model_name.replace('/', '-')}_seed_{seed}"

            #wandb.init(project='anlp_ex1', entity='amitai-edrei',
            #          name=exp_name, config={'model_name': model_name, 'seed': seed},
            #          reinit=True)

            training_args = TrainingArguments(output_dir=os.path.join(output_dir, exp_name),
                                              # report_to='wandb',
                                              run_name=exp_name,
                                              save_strategy='no',
                                              seed=seed)

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokened_datasets['train'],
                eval_dataset=tokened_datasets['validation'],
                compute_metrics=compute_metrics,
                tokenizer=tokenizer
            )
            train_result = trainer.train()
            train_time += train_result.metrics['train_runtime']
            eval_result = trainer.evaluate(eval_dataset=tokened_datasets['validation'])
            model_res.append(eval_result['eval_accuracy'])

            training_data.append(dict(exp_name=exp_name, model_name=model_name,
                                      trainer=trainer, seed=seed,
                                      eval_accuracy=eval_result['eval_accuracy']))

            # wandb.finish()

        stats[model_name] = {'mean': np.array(model_res).mean(), 'std': np.array(model_res).std()}

    best_model_data = select_best_model_data(training_data, stats)
    predict_time = predict_and_save(best_model_data['model_name'], best_model_data['trainer'],
                                    sliced_datasets['test'])
    write_stats(stats, predict_time, train_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('num_seeds', type=int, default=3)
    parser.add_argument('num_train_samp', type=int, default=-1)
    parser.add_argument('num_val_samp', type=int, default=-1)
    parser.add_argument('num_test_samp', type=int, default=-1)
    args = parser.parse_args()
    main(args=args)
