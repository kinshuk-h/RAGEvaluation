import math
import inspect
import functools
import itertools

import ipywidgets

try:
    import seaborn
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

import numpy
import pandas
import matplotlib
import scipy.stats
from matplotlib import pyplot

def with_visualization_params(function):
    signature = inspect.signature(function)
    parameters = list(signature.parameters.values())
    parameters = [
        *parameters[:-1],
        *[
            inspect.Parameter('use_seaborn', default=True, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter('figsize', default=(16, 36), kind=inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ],
        *parameters[-1:]
    ]
    signature = signature.replace(parameters=parameters)

    @functools.wraps(function)
    def visualization_impl(*args, use_seaborn=True, figsize=(16, 36), **kwargs):
        if HAS_SEABORN and use_seaborn:
            seaborn.set_theme(rc={ 'figure.figsize': figsize })
        else:
            matplotlib.rc_file_defaults()
            pyplot.figure(figsize=figsize)

        kwargs['use_seaborn'] = use_seaborn and HAS_SEABORN
        return function(*args, **kwargs)

    visualization_impl.__signature__ = signature
    return visualization_impl

@with_visualization_params
def plot_raw_trends(results, **_):
    for i, (model_key, model_results) in enumerate(results.items(), 0):
        pyplot.subplot(6, 2, 2*i + 1)
        pyplot.title(f"Change in entropy with Number of Clues for {model_key}")
        pyplot.xlabel("Number of Clues")
        pyplot.ylabel("entropy of Correct Prediction")
        for sub_scores in model_results.values():
            sub_score_list = [ score['simple']['entropy'] for score in sub_scores.values() ]
            sub_score_idxs = [ int(k) for k in sub_scores.keys() ]
            pyplot.plot(sub_score_idxs, sub_score_list)
        pyplot.subplot(6, 2, 2*i + 2)
        pyplot.title(f"Change in Entropy with Number of Clues for {model_key}")
        pyplot.xlabel("Number of Clues")
        pyplot.ylabel("Entropy")
        for sub_scores in model_results.values():
            sub_score_list = [ score['simple']['entropy'] for score in sub_scores.values() ]
            sub_score_idxs = [ int(k) for k in sub_scores.keys() ]
            pyplot.plot(sub_score_idxs, sub_score_list)

def get_accuracies(results, res_type='simple', debug=False):
    model_accuracies = {}

    for i, (model_key, model_results) in enumerate(results.items(), 0):
        accuracies = { i: 0 for i in range(1, 6) }
        if debug:
            print(">", model_key)
        for sub_result in model_results.values():
            for si, data in sub_result.items():
                accuracies[int(si)] += data[res_type]['match']
        if debug:
            print("  Accuracies:", accuracies)
            print()

        model_accuracies[model_key] = accuracies

    return model_accuracies

GROUP_BY_FUNCTIONS = {
    'nothing'            : lambda key, num_clues: 'total',
    'clue_count'         : lambda key, num_clues: int(num_clues),
    'relation'           : lambda key, num_clues: key.rsplit(':', maxsplit=2)[1],
    'clue_count+relation': lambda key, num_clues: (int(num_clues), key.rsplit(':', maxsplit=2)[1])
}

META_GROUP_BY_FUNCTIONS = {
    'nothing'            : lambda group: 'total',
    'clue_count'         : lambda group: group[0],
    'relation'           : lambda group: group[1],
}

REDUCTION_FUNCTIONS = {
    'mean': numpy.mean,
    'sum' : numpy.sum,
    'none': lambda x: x
}

def get_accuracies_per_model(results, model, group_by='clue_count', res_type='simple'):
    model_accuracies = {}
    grouper_func = GROUP_BY_FUNCTIONS[group_by] if isinstance(group_by, str) else group_by

    for key, sub_result in results[model].items():
        for si, data in sub_result.items():
            group_id = grouper_func(key, si)
            if group_id not in model_accuracies:
                model_accuracies[group_id] = [0, 0]
            model_accuracies[group_id][0] += data[res_type]['match']
            model_accuracies[group_id][1] += 1

    return model_accuracies

def merge_accuracy_groups(accuracies, group_by='relation', reduction='mean'):
    meta_accuracies = {}
    grouper_func = META_GROUP_BY_FUNCTIONS[group_by] if isinstance(group_by, str) else group_by
    reducer_func = REDUCTION_FUNCTIONS[reduction] if isinstance(reduction, str) else reduction

    for group, (match, total) in accuracies.items():
        group_id = grouper_func(group)
        if group_id not in meta_accuracies:
            meta_accuracies[group_id] = [[], []]
        meta_accuracies[group_id][0].append(match)
        meta_accuracies[group_id][1].append(total)

    return {
        group: [ reducer_func(matches), reducer_func(totals) ]
        for group, (matches, totals) in meta_accuracies.items()
    }

@with_visualization_params
def plot_accuracies(results, result_type='simple', zoom=False, **_):
    model_accuracies = get_accuracies(results, res_type=result_type)

    acc_trend_df = pandas.DataFrame.from_records(list(model_accuracies.values()), index=list(model_accuracies.keys()))
    acc_trend_df = acc_trend_df.reset_index().rename(columns={ 'index': 'model' })
    acc_trend_df = acc_trend_df.melt(id_vars='model').rename(columns=str.title).rename(columns={ 'Variable': 'Num Clues' })
    acc_trend_df['Percentage'] = acc_trend_df.apply(lambda x: round(x['Value'] / len(next(iter(results.values()))) * 100, 2), axis=1)

    if not zoom: pyplot.ylim(0, 100)

    if HAS_SEABORN:
        ax = seaborn.barplot(x='Model', y='Percentage', hue='Num Clues', data=acc_trend_df)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f %%', fontsize=10)

def get_delta_accuracies(results, res_type='simple', debug=False):
    model_delta_accuracies = {}

    for i, (model_key, model_results) in enumerate(results.items(), 0):
        delta_accuracies = { key: 0 for i in range(1, 6) for key in (f"{i}+", f"{i}-") }
        known_subjects = set()
        for i in range(1, 6):
            new_known_subjects = set(
                sub for sub, sub_result in model_results.items()
                if sub_result[str(i)][res_type]['match'] == 1
            )
            common_subjects = (new_known_subjects & known_subjects)
            delta_accuracies[f"{i}+"] = len(new_known_subjects) - len(common_subjects)
            delta_accuracies[f"{i}-"] = len(known_subjects) - len(common_subjects)
            known_subjects = new_known_subjects
        if debug:
            print(">", model_key)
            print("  Accuracies:", delta_accuracies)
            print()

        model_delta_accuracies[model_key] = delta_accuracies

    return model_delta_accuracies

@with_visualization_params
def plot_delta_accuracies(results, **_):
    model_delta_accuracies = get_delta_accuracies(results)

    acc_trend_df = pandas.DataFrame.from_records(list(model_delta_accuracies.values()), index=list(model_delta_accuracies.keys()))
    acc_trend_df = acc_trend_df.reset_index().rename(columns={ 'index': 'model' })
    acc_trend_df = acc_trend_df.melt(id_vars='model').rename(columns=str.title).rename(columns={ 'Variable': 'Num Clues' })
    # acc_trend_df['Percentage'] = acc_trend_df.apply(lambda x: round(x['Value'], 2), axis=1)

    for i, (model_key, delta_accuracies) in enumerate(model_delta_accuracies.items()):
        pyplot.subplot(3, 2, i+1)
        del_acc = { str(i): { '+': 0, '-': 0 } for i in range(1, 6) }
        for key, acc in delta_accuracies.items(): del_acc[key[:-1]][key[-1]] = abs(acc)
        acc_trend_df = pandas.DataFrame.from_records(list(del_acc.values()), index=list(del_acc.keys()))
        acc_trend_df = acc_trend_df.reset_index().rename(columns={ 'index': 'Clues' })
        acc_trend_df = acc_trend_df.melt(id_vars='Clues').rename(columns={ 'variable': 'Delta', 'value': 'Accuracy' })
        pyplot.title(model_key)
        if HAS_SEABORN:
            ax = seaborn.barplot(x='Clues', y='Accuracy', hue='Delta', data=acc_trend_df)
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', fontsize=10)

def get_trend(sequence):
    if all(numpy.greater_equal(next_value, value) for value, next_value in itertools.pairwise(sequence)):
        return 'monotone-increasing'
    if all(numpy.less_equal(next_value, value) for value, next_value in itertools.pairwise(sequence)):
        return 'monotone-decreasing'
    if all(numpy.equal(next_value, value) for value, next_value in itertools.pairwise(sequence)):
        return 'constant'
    sequence, max_idx, min_idx = list(sequence), numpy.argmax(sequence), numpy.argmin(sequence)
    if max_idx != 0 and max_idx != len(sequence)-1:
        if get_trend(sequence[:max_idx]) == 'monotone-increasing' and get_trend(sequence[max_idx+1:]) == 'monotone-decreasing':
            return 'hill'
    if min_idx != 0 and min_idx != len(sequence)-1:
        if get_trend(sequence[:min_idx]) == 'monotone-decreasing' and get_trend(sequence[min_idx+1:]) == 'monotone-increasing':
            return 'valley'
    return 'random'

def get_trends(results, res_type='simple', debug=False):
    model_trends = {}

    for i, (model_key, model_results) in enumerate(results.items(), 0):
        trends = {
            'probability': {
                'monotone-increasing': 0,
                'monotone-decreasing': 0,
                'constant'           : 0,
                'hill'               : 0,
                'valley'             : 0,
                'random'             : 0,
            },
            'entropy': {
                'monotone-increasing': 0,
                'monotone-decreasing': 0,
                'constant'           : 0,
                'hill'               : 0,
                'valley'             : 0,
                'random'             : 0,
            }
        }
        for idx, sub_scores in enumerate(model_results.values()):
            sub_score_list = [ score[res_type]['probability'] for score in sub_scores.values() ]
            trends['probability'][get_trend(sub_score_list)] += 1
        for idx, sub_scores in enumerate(model_results.values()):
            sub_score_list = [ score[res_type]['entropy'] for score in sub_scores.values() ]
            trends['entropy'][get_trend(sub_score_list)] += 1
        if debug:
            print(">", model_key)
            print("  Probability trend:", trends['probability'])
            print("  Entropy     trend:", trends['entropy'])
            print()

        model_trends[model_key] = trends

    return model_trends

@with_visualization_params
def plot_trends(results, value_type='probability', result_type='simple', zoom=True, **_):
    model_trends = get_trends(results, res_type=result_type)

    trend_df = pandas.DataFrame.from_records([ trend[value_type] for trend in model_trends.values() ], index=list(model_trends.keys()))
    trend_df = trend_df.reset_index().rename(columns={ 'index': 'model' })
    trend_df = trend_df.melt(id_vars='model').rename(columns=str.title).rename(columns={ 'Variable': 'Trend' })
    trend_df['Percentage'] = trend_df.apply(lambda x: round(x['Value'] / len(next(iter(results.values()))) * 100, 2), axis=1)

    if not zoom: pyplot.ylim(0, 100)

    if HAS_SEABORN:
        ax = seaborn.barplot(x='Model', y='Percentage', hue='Trend', data=trend_df)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f %%', fontsize=10)

def compute_confidence_interval(data: numpy.ndarray, percentage=0.95):
    return numpy.array([
        scipy.stats.t.interval(
            percentage, len(row)-1, loc=row.mean(), scale=scipy.stats.sem(row)
        )[1] - row.mean()
        for row in data
    ])

@with_visualization_params
def plot_accuracy_trends(
    results, l0=True, l1=True, l2=True, l3=False,
    error_type='ci_95', zoom=False, scatter=False, **_
):
    results, expr_details = results
    level_bools = [ l0, l1, l2, l3 ]
    for idx, (level, data) in enumerate(expr_details.items()):
        if not level_bools[idx]: continue
        acc_values = [
            get_accuracies(results[level][i], res_type='simple', debug=False)
            for i in range(len(results['L0']))
        ]
        if scatter:
            for sample_acc in acc_values:
                for i, model_key in enumerate(results['L0'][0]):
                    pyplot.subplot(math.ceil(len(results['L0'][0]) / 2), 2, i+1)
                    pyplot.title(model_key)
                    if not zoom:
                        pyplot.ylim(0, 100)
                        pyplot.yticks(numpy.linspace(10, 100, 10))
                    pyplot.xticks(numpy.linspace(1, 5, 5))
                    sample_model_acc = { key: value / len(next(iter(results['L0'][0].values()))) * 100 for key, value in sample_acc[model_key].items() }
                    pyplot.scatter(sample_model_acc.keys(), sample_model_acc.values(), label=data['label'], c=data['color'])
                    pyplot.xlabel("Number of Clues")
                    pyplot.ylabel("Accuracy (%)")
        else:
            accuracies = {
                model_key: {
                    key: [ acc_values[i][model_key][key] for i in range(len(results['L0'])) ]
                    for key in acc_values[0][model_key]
                }
                for model_key in acc_values[0]
            }
            for i, model_key in enumerate(results['L0'][0]):
                pyplot.subplot(math.ceil(len(results['L0'][0]) / 2), 2, i+1)
                pyplot.title(model_key)
                if not zoom:
                    pyplot.ylim(0, 100)
                    pyplot.yticks(numpy.linspace(10, 100, 10))
                pyplot.xticks(numpy.linspace(1, 5, 5))
                model_accuracy = {
                    key: [value / len(next(iter(results['L0'][0].values()))) * 100 for value in values]
                    for key, values in accuracies[model_key].items()
                }
                acc_mean = numpy.array(list(model_accuracy.values())).mean(axis=1)
                if error_type == 'stddev':
                    acc_err  = numpy.array(list(model_accuracy.values())).std(axis=1)
                elif error_type.startswith('ci'):
                    acc_err = compute_confidence_interval(numpy.array(list(model_accuracy.values())), int(error_type[3:])/100)
                pyplot.plot(model_accuracy.keys(), acc_mean, 'o-', label=data['label'], c=data['color'])
                pyplot.fill_between(model_accuracy.keys(), acc_mean-acc_err, acc_mean+acc_err, color=data['color'], alpha=0.4)
                pyplot.xlabel("Number of Clues")
                pyplot.ylabel("Accuracy (%)")
                pyplot.legend()

@with_visualization_params
def plot_accuracy_trends_per_relation(
    results, model, error_type='ci_95', zoom=False,
    l0=True, l1=True, l2=True, l3=False, **_
):
    results, expr_details = results
    dumped_res_map = False
    level_bools = [ l0, l1, l2, l3 ]
    margin, index, total = 0.1, 0, sum(1 if level else 0 for level in level_bools)
    # offset = ((total-1) if total & 1 else total) >> 1
    width = (1. - 2.*margin) / total

    for idx, (level, data) in enumerate(expr_details.items()):
        if not level_bools[idx]: continue

        accuracies = [
            merge_accuracy_groups(get_accuracies_per_model(
                results[level][i], model, 'clue_count+relation'
            ))
            for i in range(len(results['L0']))
        ]

        rel_keys = set().union(*[ set(sample_accuracies.keys()) for sample_accuracies in accuracies ])
        rel_map  = { rel: f'R{i}' for i, rel in enumerate(rel_keys, 1) }
        if not dumped_res_map:
            for rel, ref in rel_map.items():
                print(f"{ref:<4}", ":", rel)
            dumped_res_map = True

        aggregate_accuracies = merge_accuracy_groups(
            {
                (sample, rel_map[relation]): sample_rel_acc
                for sample, sample_acc in enumerate(accuracies)
                for relation, sample_rel_acc in sample_acc.items()
            },
            group_by=lambda x: x[1],
            reduction='none'
        )

        agg_acc_vals = [
            [ match / total * 100 for match, total in zip(matches, totals) ]
            for matches, totals in aggregate_accuracies.values()
        ]

        # agg_acc_match_vals = [
        #     [ match for match, total in zip(matches, totals) ]
        #     for matches, totals in aggregate_accuracies.values()
        # ]
        agg_acc_total_vals = [
            [ total for match, total in zip(matches, totals) ]
            for matches, totals in aggregate_accuracies.values()
        ]

        # agg_match_mean = numpy.mean(agg_acc_match_vals, axis=1)
        agg_total_mean = numpy.mean(agg_acc_total_vals, axis=1)

        acc_mean = numpy.array(agg_acc_vals).mean(axis=1)
        if error_type == 'stddev':
            acc_err  = numpy.array(list(agg_acc_vals)).std(axis=1)
        elif error_type.startswith('ci'):
            acc_err = compute_confidence_interval(numpy.array(list(agg_acc_vals)), int(error_type[3:])/100)

        # labels = [
        #     f"({match:.2f} / {total:.2f}) ± {err:.2f}%"
        #     for match, total, err in zip(agg_match_mean, agg_total_mean, acc_err)
        # ]

        if not zoom:
            pyplot.ylim(0, 100)
            pyplot.yticks(numpy.linspace(10, 100, 10))

        xticks = numpy.arange(len(aggregate_accuracies)) + margin + (index * width)

        pyplot.xticks(xticks - margin - (index * width) + 0.5, labels=[
            f"{ref} ({int(tot)})" for ref, tot in zip(rel_map.values(), agg_total_mean)
        ], rotation=90)
        rects = pyplot.bar(
            xticks, acc_mean, width=width, align='edge',
            yerr=acc_err, label=data['label'], color=data['color']
        )
        # pyplot.bar_label(rects, labels, fmt="%.1f", rotation=90)
        pyplot.xlabel("Relations")
        pyplot.ylabel("Accuracy (%)")
        pyplot.legend()

        index += 1

@with_visualization_params
def plot_prob_mass_trends(results, **_):
    margin = 0.1

    for i, model_key in enumerate(results):
        pyplot.subplot(math.ceil(len(results) / 2), 2, i+1)
        pyplot.title(model_key)

        width = (1. - 2.*margin) / len(results[model_key])
        labels = [ key.upper() for key in results[model_key].keys() ]
        xticks = numpy.arange(len(results[model_key])) + margin

        token_prob = numpy.mean([
            [
                (
                    max(value['probability']['token'].values()) if value['match']['token']
                    else min(value['probability']['token'].values())
                )
                for value in values.values()
            ]
            for values in results[model_key].values()
        ], axis=1)

        concept_prob = numpy.mean([
            [
                (
                    max(value['probability']['concept'].values()) if value['match']['concept']
                    else min(value['probability']['concept'].values())
                )
                for value in values.values()
            ]
            for values in results[model_key].values()
        ], axis=1)

        pyplot.bar(xticks, token_prob, label='token', width=width, align='edge')
        pyplot.bar(
            xticks + width, concept_prob,
            label='concept', width=width, align='edge'
        )
        pyplot.xticks(xticks + width, labels=labels)
        pyplot.ylim(0, 1)

        pyplot.xlabel("Type")
        pyplot.ylabel("Probability Mass")
        pyplot.legend()

@with_visualization_params
def plot_validation_trends(results, **_):
    model_keys = sorted(set(map(lambda x: x.split('->')[0], results.data().keys())))
    margin = 0.1

    # figure, axes = pyplot.subplots(len(model_keys), len(model_keys))

    # for ax, model_key in zip(axes[0], model_keys):
    #     ax.set_title(model_key)

    # for ax in axes[-1]:
    #     ax.set_xlabel('')

    # for ax, model_key in zip(axes[:, 0], model_keys):
    #     ax.set_ylabel(model_key, rotation=0, size='large')

    # figure.tight_layout()

    for i, eval_model in enumerate(model_keys):
        pyplot.subplot(1, len(model_keys), i+1)
        width = (1. - 2.*margin) / len(model_keys)
        labels = list(model_keys)
        xticks = numpy.arange(len(model_keys)) + margin

        token_acc = numpy.mean([
            [
                numpy.mean([
                    1 if claim['validation']['match']['token'] else 0
                    for segment in bio for claim in segment
                ])
                for bio in results[f"{gen_model}->{eval_model}"].values()
            ]
            for gen_model in model_keys
        ], axis=1)

        concept_acc = numpy.mean([
            [
                numpy.mean([
                    1 if claim['validation']['match']['concept'] else 0
                    for segment in bio for claim in segment
                ])
                for bio in results[f"{gen_model}->{eval_model}"].values()
            ]
            for gen_model in model_keys
        ], axis=1)

        pyplot.bar(xticks, token_acc, label='token', width=width, align='edge')
        pyplot.bar(
            xticks + width, concept_acc,
            label='concept', width=width, align='edge'
        )
        pyplot.xticks(xticks + width, labels=labels, rotation=45)
        pyplot.ylim(0, 1)

        pyplot.title(f"Evaluation w/ {eval_model}")
        if i == 0:
            pyplot.ylabel("Acceptance")
        pyplot.legend()

@with_visualization_params
def plot_validation_inference_trends(results, popularity='high', level='concept', **_):
    model_keys = sorted(set(map(lambda x: x.split('->')[0], results[next(iter(results.keys()))][popularity].data().keys())))
    margin = 0.1

    # figure, axes = pyplot.subplots(len(model_keys), len(model_keys))

    # for ax, model_key in zip(axes[0], model_keys):
    #     ax.set_title(model_key)

    # for ax in axes[-1]:
    #     ax.set_xlabel('')

    # for ax, model_key in zip(axes[:, 0], model_keys):
    #     ax.set_ylabel(model_key, rotation=0, size='large')

    # figure.tight_layout()

    for i, eval_model in enumerate(model_keys):
        pyplot.subplot(len(model_keys), 1, i+1)
        width = (1. - 2.*margin) / len(results)
        labels = list(model_keys)
        xticks = numpy.arange(len(model_keys)) + margin

        for index, (inference_type, type_results) in enumerate(results.items()):
            level_acc = numpy.mean([
                [
                    numpy.mean([
                        1 if claim['validation']['match'][level] else 0
                        for segment in bio for claim in segment
                    ])
                    for bio in type_results[popularity][f"{gen_model}->{eval_model}"].values()
                ]
                for gen_model in model_keys
            ], axis=1)

            pyplot.bar(xticks + (index * width), level_acc, label=inference_type, align='edge', width=width)

        pyplot.xticks(xticks + (len(results) >> 1) * width, labels=labels, rotation=0)
        if i == len(model_keys) - 1:
            pyplot.xlabel("Generator Model")
        pyplot.ylabel("Acceptance Ratio")
        pyplot.ylim(0, 1)

        pyplot.title(f"Evaluation w/ {eval_model}")
        pyplot.legend()

@with_visualization_params
def plot_validation_histograms(results, bins=10, negative=False, level='token', **_):

    def validate_token_claim(claim):
        true_pos_prob = claim['probability']['token']['positive']
        true_neg_prob = claim['probability']['token']['negative']
        return true_pos_prob > true_neg_prob
    def validate_concept_claim(claim):
        return claim['probability']['concept']['positive'] > claim['probability']['concept']['negative']

    validation = {
        'token'  : validate_token_claim,
        'concept': validate_concept_claim
    }

    model_keys = sorted(set(map(lambda x: x.split('->')[0], results.data().keys())))
    for i, eval_model in enumerate(model_keys):
        for j, gen_model in enumerate(model_keys):
            pyplot.subplot(len(model_keys), len(model_keys), i*len(model_keys)+j+1)

            data = [
                numpy.mean([
                    1 if (negative ^ validation[level](claim['validation'])) else 0
                    for segment in bio for claim in segment
                ])
                for bio in results[f"{gen_model}->{eval_model}"].values()
            ]

            if _.get('use_seaborn', False):
                seaborn.histplot(x=data, bins=bins, binrange=(0, 1))
            else:
                pyplot.hist(data, bins=bins, range=(0, 1))

            if i == 0:
                if j == (len(model_keys) >> 1): pyplot.title('Generator Model\n'+gen_model)
                else: pyplot.title(gen_model)
            if j == 0:
                if i == (len(model_keys) >> 1): pyplot.ylabel('Evaluation Model\n'+eval_model)
                else: pyplot.ylabel(eval_model)
            if j == (len(model_keys) >> 1) and i == len(model_keys)-1:
                pyplot.xlabel(f"Ratio of {'Una' if negative else 'A'}ccepted Claims across Biographies")

@with_visualization_params
def plot_prob_mass_ratios(results, popularity='high', level='concept', **_):
    model_keys = sorted(set(map(lambda x: x.split('->')[0], results[next(iter(results.keys()))][popularity].data().keys())))
    margin = 0.1

    for i, eval_model in enumerate(model_keys):
        pyplot.subplot(len(model_keys), 1, i+1)
        width = (1. - 2.*margin) / len(results)
        labels = list(model_keys)
        xticks = numpy.arange(len(model_keys)) + margin

        for index, (inference_type, type_results) in enumerate(results.items()):
            avg_mass = numpy.mean([
                [
                    numpy.mean([
                        claim['validation']['probability'][level]['positive'] + claim['validation']['probability'][level]['negative']
                        for segment in bio for claim in segment
                    ])
                    for bio in type_results[popularity][f"{gen_model}->{eval_model}"].values()
                ]
                for gen_model in model_keys
            ], axis=1)
            avg_mass_pos = numpy.mean([
                [
                    numpy.mean([
                        claim['validation']['probability'][level]['positive']
                        for segment in bio for claim in segment
                    ])
                    for bio in type_results[popularity][f"{gen_model}->{eval_model}"].values()
                ]
                for gen_model in model_keys
            ], axis=1)

            bars = pyplot.bar(xticks + (index * width), avg_mass, label=inference_type, align='edge', width=width)
            color = bars.patches[0].get_facecolor()
            d_color = ( *[ c * 0.5 for c in color[:-1] ], color[-1] )
            pyplot.bar(xticks + (index * width), avg_mass, color=d_color, align='edge', width=width)
            pyplot.bar(xticks + (index * width), avg_mass_pos, color=color, align='edge', width=width)

        pyplot.xticks(xticks + (len(results) >> 1) * width, labels=labels, rotation=0)
        if i == len(model_keys) - 1:
            pyplot.xlabel("Generator Model")
        pyplot.ylabel("Probability")
        # pyplot.ylim(0, 1)

        pyplot.title(f"Evaluation w/ {eval_model}")
        pyplot.legend()

def compute_data_for_bio_task(bio_data, pred_data, level='token'):
    data = [ 0, 0, 0, 0 ]
    for entity in bio_data:
        for bio_segment, pred_segment in zip(bio_data[entity]['decomposition'], pred_data[entity]):
                for decision, claim in zip(bio_segment['decisions'], pred_segment):
                    if claim['validation']['match'][level] == decision['is_supported']:
                        data[ 0 if decision['is_supported'] else 3 ] += 1
                    else:
                        data[ 1 if decision['is_supported'] else 2 ] += 1
    return data

def compute_data_for_qa_task(qa_data, pred_data, level='token'):
    data = [ 0, 0, 0, 0 ]
    for idx in qa_data:
        qa_segment, pred_segment = qa_data[idx], pred_data[idx]
        if pred_segment['match'][level] == qa_segment['validation']['match']:
            data[ 0 if qa_segment['validation']['match'] else 3 ] += 1
        else:
            data[ 1 if qa_segment['validation']['match'] else 2 ] += 1
    return data

TASK_COMPUTE_FUNCTIONS = {
    'bio': compute_data_for_bio_task,
    'qa' : compute_data_for_qa_task
}

@with_visualization_params
def plot_factuality_trends(results, level='concept', all_pairs=True, task_type='bio', **_):
    model_keys = sorted(set(map(lambda x: x.split('->')[0], results.keys())))

    compute_data_function = TASK_COMPUTE_FUNCTIONS[task_type or 'bio']
    
    if all_pairs:
        for i, eval_model in enumerate(model_keys):
            for j, gen_model in enumerate(model_keys):
                pyplot.subplot(len(model_keys), len(model_keys), i*len(model_keys)+j+1)

                data = compute_data_function(*results[f"{gen_model}->{eval_model}"], level=level)

                data_ratios = numpy.array(data) / numpy.sum(data)
                ticks = list(range(len(data)))
                rect = pyplot.bar(ticks, data_ratios)
                pyplot.bar_label(rect, [ f"{pt:.2f}" for pt in data_ratios ])
                pyplot.xticks(ticks, labels=[ 'TT', 'TF', 'FT', 'FF' ])
                pyplot.ylim(0, 1)

                if i == 0:
                    if j == (len(model_keys) >> 1): pyplot.title('Generator Model\n'+gen_model)
                    else: pyplot.title(gen_model)
                if j == 0:
                    if i == (len(model_keys) >> 1): pyplot.ylabel('Evaluation Model\n'+eval_model)
                    else: pyplot.ylabel(eval_model)

    else:
        for i, eval_model in enumerate(model_keys):
            gen_model = eval_model
            pyplot.subplot(1, len(model_keys), i+1)

            data = compute_data_function(*results[f"{gen_model}->{eval_model}"], level=level)

            print(f"Number of atomic claims generated by {gen_model:20}:", sum(data))

            data_ratios = numpy.array(data) / numpy.sum(data)
            ticks = list(range(len(data)))
            rect = pyplot.bar(ticks, data_ratios)
            pyplot.bar_label(rect, [ f"{pt:.2f}" for pt in data_ratios ])
            pyplot.xticks(ticks, labels=[ 'TT', 'TF', 'FT', 'FF' ])
            pyplot.ylim(0, 1)

            pyplot.title(gen_model)

@with_visualization_params
def plot_validation_trends_qa(results, **_):
    model_keys = sorted(set(map(lambda x: x.split('->')[0], results.data().keys())))
    margin = 0.1

    # figure, axes = pyplot.subplots(len(model_keys), len(model_keys))

    # for ax, model_key in zip(axes[0], model_keys):
    #     ax.set_title(model_key)

    # for ax in axes[-1]:
    #     ax.set_xlabel('')

    # for ax, model_key in zip(axes[:, 0], model_keys):
    #     ax.set_ylabel(model_key, rotation=0, size='large')

    # figure.tight_layout()

    for i, eval_model in enumerate(model_keys):
        pyplot.subplot(1, len(model_keys), i+1)
        width = (1. - 2.*margin) / len(model_keys)
        labels = list(model_keys)
        xticks = numpy.arange(len(model_keys)) + margin

        token_acc = numpy.mean([
            [
                1 if claim['match']['token'] else 0
                for claim in results[f"{gen_model}->{eval_model}"].values()
            ]
            for gen_model in model_keys
        ], axis=1)

        concept_acc = numpy.mean([
            [
                1 if claim['match']['concept'] else 0
                for claim in results[f"{gen_model}->{eval_model}"].values()
            ]
            for gen_model in model_keys
        ], axis=1)

        pyplot.bar(xticks, token_acc, label='token', width=width, align='edge')
        pyplot.bar(
            xticks + width, concept_acc,
            label='concept', width=width, align='edge'
        )
        pyplot.xticks(xticks + width, labels=labels, rotation=45)
        pyplot.ylim(0, 1)

        pyplot.title(f"Evaluation w/ {eval_model}")
        if i == 0:
            pyplot.ylabel("Acceptance")
        pyplot.legend()

@with_visualization_params
def plot_qa_answer_frequency_trends(results, subset, **_):

    margin = 0.1
    model_keys = sorted(set(map(lambda x: x.split('->')[0], results[0][subset].data().keys())))

    for i, eval_model in enumerate(model_keys):
        for j, gen_model in enumerate(model_keys):
            pyplot.subplot(len(model_keys), len(model_keys), i*len(model_keys)+j+1)

            pyplot.xticks(list(results.keys()))

            width = (1. - 2.*margin) / len(model_keys)
            xticks = numpy.arange(len(results)) + margin

            token_acc = numpy.mean([
                [
                    1 if claim['match']['token'] else 0
                    for claim in results[freq][subset][f"{gen_model}->{eval_model}"].values()
                ]
                for freq in results
            ], axis=1)

            concept_acc = numpy.mean([
                [
                    1 if claim['match']['concept'] else 0
                    for claim in results[freq][subset][f"{gen_model}->{eval_model}"].values()
                ]
                for freq in results
            ], axis=1)

            pyplot.plot(
                xticks, token_acc,
                marker='o', label='token', color='tab:blue'
                # width=width, align='edge'
            )
            pyplot.plot(
                xticks, concept_acc,
                label='concept', marker='o', color='tab:orange'
                #width=width, align='edge'
            )
            # pyplot.xticks(xticks + width, labels=labels, rotation=45)
            pyplot.ylim(0, 1)

            if i == 0:
                if j == (len(model_keys) - 1): pyplot.legend()
                if j == (len(model_keys) >> 1): pyplot.title('Generator Model\n'+gen_model)
                else: pyplot.title(gen_model)
            if j == 0:
                if i == (len(model_keys) >> 1): pyplot.ylabel('Evaluation Model\n'+eval_model)
                else: pyplot.ylabel(eval_model)
            if j == (len(model_keys) >> 1) and i == len(model_keys)-1:
                pyplot.xlabel("Frequency of Greedy-decoded answer")

def interactive(vis_function, **kwargs):
    results = kwargs.get('results', None)
    if results is not None:
        del kwargs['results']

    task_type = kwargs.get('task_type', 'bio')
    if 'task_type' in kwargs:
        del kwargs['task_type']

    argspec = inspect.getfullargspec(vis_function)
    defargs = dict(zip(argspec.args[-len(argspec.defaults):], argspec.defaults))

    figsize = kwargs.get('figsize', defargs['figsize'])
    if 'figsize' in kwargs: del kwargs['figsize']

    use_seaborn = kwargs.get('use_seaborn', defargs.get('use_seaborn', True))
    if 'use_seaborn' in kwargs: del kwargs['use_seaborn']

    for arg, value in kwargs.items():
        if isinstance(value, bool):
            kwargs[arg] = ipywidgets.ToggleButton(
                value=value, icon="check",
                description=arg.replace('-', ' ').replace('_', ' ').title(),
            )
        elif isinstance(value, (tuple, list)) and isinstance(value[0], bool):
            kwargs[arg] = ipywidgets.ToggleButton(
                value=value[0], icon="check", description=value[1]
            )

    signature = inspect.signature(vis_function)

    parameters = [ param for key, param in signature.parameters.items() if key not in ('results', 'figsize', 'task_type', '_') ]
    parameters.extend([
        inspect.Parameter('scale', default=1, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ])
    kwargs.update(
        use_seaborn=ipywidgets.ToggleButton(value=use_seaborn, description='Seaborn Theme', icon='check'),
        scale=ipywidgets.FloatSlider(value=1, description="Figure Scale", min=0.2, max=2, step=0.2, continuous_update=False)
    )
    signature = signature.replace(parameters=parameters)

    def vis_function_interactive(**kwdargs):
        scale = kwdargs['scale']
        del kwdargs['scale']
        kwdargs['figsize'] = (figsize[0] * scale, figsize[1] * scale)
        if results: kwdargs['results'] = results
        if task_type: kwdargs['task_type'] = task_type
        return vis_function(**kwdargs)
    vis_function_interactive.__signature__ = signature

    # def on_save():
    #     pyplot.savefig()

    # save_button = ipywidgets.Button(icon='save')
    # save_button.on_click(on_save)

    widget = ipywidgets.interactive(
        vis_function_interactive,
        **kwargs
    )
    flex_layout = ipywidgets.Layout(
        display='flex',
        flex_flow='row wrap',
        justify_content='space-around',
        align_items='flex-start',
        width='100%'
    )
    return ipywidgets.VBox(
        [
            ipywidgets.Box(widget.children[:-3], layout=flex_layout),
            ipywidgets.HBox([
                *widget.children[-3:-1],
                # save_button,
            ], layout=flex_layout),
            widget.children[-1],
            # ipywidgets.HBox([ widget.children[-1] ], layout=flex_layout)
        ]
    )
