import json

import numpy
import pandas
import datasets

from tqdm.auto import tqdm

dedup_df, sub_rel_groups, obj_rel_groups = None, None, None

def get_obj_rows(entry=None, object=None, exclude_if_object=None):
    global dedup_df, sub_rel_groups, obj_rel_groups

    assert object is not None or entry is not None

    object = object or entry['obj_label_icase']
    as_object = set(obj_rel_groups.groups[object])
    as_subject = set(sub_rel_groups.groups.get(object, []))
    if exclude_if_object:
        as_subject = set(
            row for row in as_subject
            if dedup_df.loc[row, 'obj_label_icase'] != exclude_if_object
        )
    obj_rows = dedup_df.iloc[list(as_object | as_subject), :]
    if entry is not None:
        obj_rows = obj_rows[obj_rows['uuid'] != entry['uuid']]

    return obj_rows

def __prepare_dataset():
    global dedup_df, sub_rel_groups, obj_rel_groups
    # fact_dataset = datasets.load_from_disk("data/lama-dedup-w-qa")
    fact_dataset = datasets.load_dataset("lama")['train']
    augmentation_dataset = datasets.load_dataset("janck/bigscience-lama")['test'].to_pandas()[['uuid', 'question']]

    dedup_df = fact_dataset.to_pandas()
    dedup_df = dedup_df.drop_duplicates(
        subset=[ 'sub_uri', 'predicate_id', 'obj_uri' ], ignore_index=True
    )
    dedup_df = dedup_df.merge(augmentation_dataset, on='uuid')
    dedup_df = dedup_df.assign(
        sub_label_icase = dedup_df.sub_label.apply(lambda x: x.lower()),
        obj_label_icase = dedup_df.obj_label.apply(lambda x: x.lower()),
    )

    sub_rel_groups = dedup_df.groupby([ 'sub_label_icase' ])
    # subjects = sub_rel_groups.count()['uuid'].nlargest(200).index

    obj_rel_groups = dedup_df.groupby([ 'obj_label_icase' ])
    # objects = obj_rel_groups.count()['uuid'].nlargest(200).index

    # Filter out 1-1 and N-1 relation based triples
    working_df = dedup_df[dedup_df['type'].isin([ '1-1', 'N-1' ])]

    # Filter out rows with less than 5 clues
    idx = []
    for _, entry in tqdm(working_df.iterrows(), total=len(working_df)):
        if len(get_obj_rows(entry)) > 4: idx.append(entry['uuid'])

    working_df = working_df[working_df['uuid'].isin(idx)]

    # Filter out 1-N relations (ambiguously marked as N-1)
    tmp_grp = working_df.groupby([ 'label', 'sub_label' ])
    useful_idx = pandas.Index([], dtype=next(iter(tmp_grp.groups.values())).dtype)
    for grp_ref, idx in tmp_grp.groups.items():
        if len(idx) == 1: useful_idx = useful_idx.union(idx)

    working_df = working_df.loc[useful_idx]

    obj_groups = working_df.groupby('obj_label_icase')
    obj_dict = {
        obj: obj_grp['predicate_id'].unique()
        for obj, obj_grp in obj_groups
    }

    return working_df, obj_dict

def __prepare_contexts():
    with open('data/lama.trex_dedup.artificial.wiki.context.json', 'r+', encoding='utf-8') as ifile:
        artificial_contexts = json.load(ifile)

    return {
        'artificial': artificial_contexts
    }

contexts             = __prepare_contexts()
working_df, obj_dict = __prepare_dataset()

def get_dataset():
    global working_df
    return working_df

def generate_working_samples(samples=5, size=None, random_state=None):
    df_samples = []
    global working_df, contexts
    current_working_df = working_df
    for i in range(samples):

        obj_groups = current_working_df.groupby('obj_label_icase')
        obj_counts = obj_groups['uuid'].count()

        random_seed = (random_state + i if isinstance(random_state, int) else random_state)

        sample = (
            obj_groups
            .sample(n=1, random_state=random_seed)
            .sample(n=size or len(obj_groups.groups), random_state=random_state)
        )
        uuids = [ row.uuid for _, row in sample.iterrows() if obj_counts[row.obj_label_icase] > 1 ]

        if len(uuids) != 0:
            current_working_df = current_working_df[~current_working_df.uuid.isin(uuids)]

        df_samples.append({
            'data': sample,
            'seed': random_seed
        })
    return df_samples

def get_dissimilar_entities(relation_list, count=5):
    ob_score = {
        ob: (
            len(numpy.intersect1d(ob_relations, relation_list, assume_unique=True)),
            -len(ob_relations)
        )
        for ob, ob_relations in obj_dict.items()
    }
    return sorted(ob_score.keys(), key=lambda x: ob_score[x])[:count]

def get_most_similar_entity(object, relation_list):
    sim_object, best_count = None, 0
    for obj, obj_relations in obj_dict.items():
        if obj == object: continue
        count = len(numpy.intersect1d(obj_relations, relation_list, assume_unique=True))
        if count > best_count: sim_object, best_count = obj, count
        elif count > 0 and count == best_count:
            obj_rel = set(relation_list)
            if len(set(obj_dict[obj]) - obj_rel) <= len(set(obj_dict[sim_object]) - obj_rel):
            # if len(dedup_df[dedup_df['sub_label'] == sub]) > len(dedup_df[dedup_df['sub_label'] == sim_subject]):
                sim_object, best_count = obj, count
    return sim_object

def sample_diverse_rows(object, obj_rows, max_n_clues, random_state=None):
    uniq_rel_obj_rows = obj_rows.groupby('predicate_id')
    if len(uniq_rel_obj_rows.groups) >= max_n_clues:
        rows = uniq_rel_obj_rows.sample(n=1, random_state=random_state)\
                .sample(n=max_n_clues, random_state=random_state)
    else:
        if len(obj_rows) > max_n_clues:
            as_sub_rows = obj_rows[obj_rows['sub_label_icase'] == object]
            as_obj_rows = obj_rows[obj_rows['obj_label_icase'] == object]
            half_size = max_n_clues // 2
            if len(as_sub_rows) <= half_size:
                rows = as_obj_rows.sample(n=max_n_clues-len(as_sub_rows), random_state=random_state)
                rows = pandas.concat([ rows, as_sub_rows ], axis=0)
            elif len(as_obj_rows) <= half_size:
                rows = as_sub_rows.sample(n=max_n_clues-len(as_obj_rows), random_state=random_state)
                rows = pandas.concat([ rows, as_sub_rows ], axis=0)
            else:
                rows1 = as_obj_rows.sample(n=half_size, random_state=random_state)
                rows2 = as_sub_rows.sample(n=max_n_clues-half_size, random_state=random_state)
                rows = pandas.concat([ rows1, rows2 ], axis=0)
            # rows = obj_rows.sample(n=max_n_clues, random_state=random_state)
            rows = rows.sample(frac=1, random_state=random_state)
        else:
            assert len(obj_rows) == max_n_clues
            rows = obj_rows
    return [ row for _, row in rows.iterrows() ]

def make_query(entry, how='cloze', sub_mask="[MASK]"):
    match how:
        case 'cloze'    : return entry['template'].replace('[X]', entry['sub_label']).replace('[Y]', sub_mask)[:-2]
        case 'question' : return entry['question'].replace('[X]', entry['sub_label'])

def format_clues(clues, clue_format, simple=True):
    if clue_format == 'statement':
        if simple:
            return [
                entry.template.replace('[X]', entry.sub_label.strip())
                    .replace('[Y]', entry.obj_label.strip()).strip()[:-2]
                for entry in clues
            ]
        else:
            return [
                entry.masked_sentence.replace('[MASK]', entry.obj_surface)
                for entry in clues
            ]
    elif clue_format == 'question-answer':
        return [
            (entry.question.replace('[X]', entry.sub_label), entry.obj_label)
            for entry in clues
        ]
    elif clue_format.endswith('-context'):
        num_clues, ctx_type = len(clues) - 1, clue_format.split('-')[0]
        return contexts[ctx_type][clues[-1].obj_label][num_clues][clues[-1][f'obj_ctx_{ctx_type}']]
    else:
        return []

def augment_fields(entries, sample):
    for ctx_type, context_data in contexts.items():
        engine = numpy.random.default_rng(seed=sample['seed'])

        for row in entries:
            min_size = min(len(data) for data in context_data[row.obj_label])
            row[f'obj_ctx_{ctx_type}'] = engine.integers(min_size)

    return entries

def get_supportive_clues(entry, max_n_clues=5, random_state=None):
    return sample_diverse_rows(entry['obj_label_icase'], get_obj_rows(entry), max_n_clues, random_state=random_state)

def get_random_clues(entry, max_n_clues=5, random_state=None):
    global dedup_df, obj_dict, obj_rel_groups
    unrelated_objects = get_dissimilar_entities(obj_dict[entry['obj_label_icase']])
    obj_rows = dedup_df.iloc[[ idx for obj in unrelated_objects for idx in obj_rel_groups.groups[obj] ], :]
    return sample_diverse_rows(entry['obj_label_icase'], obj_rows, max_n_clues, random_state=random_state)

def get_distracting_clues(entry, max_n_clues=5, random_state=None):
    faux_object = get_most_similar_entity(entry['obj_label_icase'], obj_dict[entry['obj_label_icase']])
    obj_rows = get_obj_rows(object=faux_object, exclude_if_object=entry.obj_label_icase)
    return sample_diverse_rows(faux_object, obj_rows, max_n_clues, random_state=random_state)

def get_no_clues(entry, max_n_clues=5, random_state=None):
    return []

def get_entry(subject, offset=0):
    assert subject in working_df['sub_label'].tolist()
    return working_df[working_df['sub_label'] == subject].iloc[offset, :]

