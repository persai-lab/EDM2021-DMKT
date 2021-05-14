from deepkt.utils.config import *
from deepkt.agents import *
import numpy as np
from sklearn import metrics
import torch
import time


def single_exp(model, data, exp_name, mode, json_config):
    config = process_config(json_config)
    agent_class = globals()[config.agent]
    agent = agent_class(config)
    agent.run()
    best_epoch, best_train_loss, best_val_perf = agent.finalize()
    config["best_epoch"] = best_epoch
    config["best_train_loss"] = best_train_loss
    config["best_val_perf"] = best_val_perf
    print(config)

    result_dir_path = "experiments/{}Agent".format(model)
    create_dirs([result_dir_path])

    if mode == "hyperparameters":
        result_file_path = "{}/{}_{}.json".format(result_dir_path, mode, data)
        if not os.path.exists(result_file_path):
            with open(result_file_path, "w") as f:
                pass
        with open(result_file_path, "a") as f:
            f.write(json.dumps(config) + "\n")
    else:
        result_file_path = "{}/{}_{}.csv".format(result_dir_path, mode, data)
        if not os.path.exists(result_file_path):
            with open(result_file_path, "w") as f:
                curr_time = time.time()
                f.write("Time: {}\n".format(curr_time))

        true_labels = torch.load(config.out_dir + "true_labels.tar")
        pred_labels = torch.load(config.out_dir + "dkvmn_pred_labels.tar")
        if config.metric == "rmse":
            rmse = np.sqrt(metrics.mean_squared_error(true_labels, pred_labels))
            mae = metrics.mean_absolute_error(true_labels, pred_labels)
            with open(result_file_path, "a") as f:
                f.write("{},{},{}\n".format(exp_name, rmse, mae))
        elif config.metric == "auc":
            roc_auc = metrics.roc_auc_score(true_labels, pred_labels)
            prec, rec, _ = metrics.precision_recall_curve(true_labels, pred_labels)
            pr_auc = metrics.auc(rec, prec)
            with open(result_file_path, "a") as f:
                f.write("{},{},{}\n".format(exp_name, roc_auc, pr_auc))
        else:
            raise AttributeError


def check_progress(model, data, args_list):
    metric = None
    result_file_dir = "experiments/{}Agent".format(model)
    result_file_path = "{}/hyperparameters_{}.json".format(result_file_dir, data)

    progress_dict = {}
    best_config = {}
    performance_dict = {}
    para_config_mapping = {}
    if not os.path.exists(result_file_path):
        create_dirs([result_file_dir])
        with open(result_file_path, "w") as f:
            pass
        return progress_dict, best_config

    with open(result_file_path, "r") as f:
        for line in f:
            try:
                result = json.loads(line)
                metric = result['metric']
            except json.decoder.JSONDecodeError:
                print(line)

            key = []
            for arg in args_list:
                key.append(result[arg])
            key = tuple(key)
            if key not in progress_dict:
                progress_dict[key] = True
            if key not in performance_dict:
                performance_dict[key] = (
                    result["best_epoch"], result["best_train_loss"], result["best_val_perf"]
                )
            if key not in para_config_mapping:
                para_config_mapping[key] = result

    print("Number of existing records: {}".format(len(progress_dict)))
    if metric is None:
        return progress_dict, best_config
    elif metric == "auc":
        sorted_perf = sorted(performance_dict.items(), key=lambda x: x[1][-1], reverse=True)
        best_train_loss = sorted_perf[0][1][1]
        best_key = sorted_perf[0][0]
        best_config = para_config_mapping[best_key]
    elif metric == "rmse":
        sorted_perf = sorted(performance_dict.items(), key=lambda x: x[1][-1])
        best_train_loss = sorted_perf[0][1][1]
        best_key = sorted_perf[0][0]
        best_config = para_config_mapping[best_key]
    else:
        raise AttributeError

    for arg in args_list:
        print("{},".format(arg), end="")
    print("best_epoch,train_loss,val_perf({})".format(metric))
    for (para, (best_epoch, train_loss, val_perf)) in sorted_perf:
        for k in para:
            print("{},".format(k), end="")
        print("{},".format(best_epoch), end="")
        print("{},".format(train_loss), end="")
        print("{}".format(val_perf))

    for i, arg in enumerate(args_list):
        best_config[arg] = sorted_perf[0][0][i]
    if "stride" in result:
        best_config.pop("stride")
    best_config["target_train_loss"] = best_train_loss  # used to early stop on training
    best_config.pop("best_epoch")
    best_config.pop("best_train_loss")
    best_config.pop("best_val_perf")
    best_config.pop("summary_dir")
    best_config.pop("checkpoint_dir")
    best_config.pop("out_dir")
    best_config.pop("log_dir")
    print(best_config)
    return progress_dict, best_config


def hyperparameters_tuning(model, data, args_list, exp_name_format, config_list, progress_dict):
    mode = "hyperparameters"
    config_dir = "configs/{}/{}".format(model, data)
    create_dirs([config_dir])
    num_exps = len(config_list)
    count = 0
    for exp_id, config_dict in enumerate(config_list):
        key = []
        for arg in args_list:
            key.append(config_dict[arg])
        key = tuple(key)
        if key not in progress_dict:
            count += 1
            exp_name = exp_name_format.format(*key)
            print("not finished exp: {}".format(exp_name))
            config_dict["exp_name"] = exp_name
            config_file_path = "{}/{}.json".format(config_dir, exp_name)
            json.dump(config_dict, open(config_file_path, "w"))
            single_exp(model, data, exp_name, mode, config_file_path)


def test_5folds(model, data, best_config):
    mode = "5folds"
    config_dir = "configs/5folds/{}/{}".format(model, data)
    create_dirs([config_dir])
    best_config["mode"] = "test"
    for fold in range(1, 6):
        print("\nfold {}".format(fold))
        best_config["data_name"] = "{}_fold_{}".format(data, fold)
        exp_name = "{}_{}_fold_{}".format(model, data, fold)
        best_config["exp_name"] = exp_name
        for key in best_config:
            print("{}: {}".format(key, best_config[key]))
        config_file_path = "{}/{}.json".format(config_dir, exp_name)
        json.dump(best_config, open(config_file_path, "w"))
        single_exp(model, data, exp_name, mode, config_file_path)
