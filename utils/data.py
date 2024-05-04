import os
import json

class Result:
    def __init__(self, file) -> None:
        self.results = {}
        self.file    = file

    def load(self):
        with open(self.file, "r+", encoding="utf-8") as ifile:
            return json.load(ifile)

    def save(self):
        with open(self.file, "w+", encoding="utf-8") as ofile:
            json.dump(self.results, ofile, ensure_ascii=False, indent=2)

    def __getitem__(self, key):
        return self.results[key]

    def __setitem__(self, key, value):
        self.results[key] = value

    def __len__(self):
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def data(self):
        return self.results

class NestedListItemResult(Result):
    def __init__(self, file, *key_lists) -> None:
        super().__init__(file)
        self.results = self.__init_or_load(*key_lists)

    @classmethod
    def recursive_create(cls, key_lists, level=0):
        if level < len(key_lists):
            return {
                key: cls.recursive_create(key_lists, level+1)
                for key in key_lists[level]
            }
        else:
            return None

    @classmethod
    def recursive_update(cls, results, key_lists, level=0):
        if level < len(key_lists):
            for key in key_lists[level]:
                if key not in results:
                    results[key] = cls.recursive_create(key_lists, level+1)
                else:
                    results[key] = cls.recursive_update(results[key], key_lists, level+1)
            return results
        else:
            return results

    def __init_or_load(self, *key_lists):
        if os.path.exists(self.file):
            results = self.load()
            if len(key_lists) > 0:
                results = self.recursive_update(results, key_lists)

        else:
            results = self.recursive_create(key_lists)

        return results

class PredictionResult(Result):
    DEFAULT_ENTRY_TO_KEY_MAPPER = lambda s, x: x['sub_label']+":"+x['label']+":"+x['obj_label']

    def __init__(self, file, working_df=None, models=None, entry_to_key_mapper=None, clue_count=5):
        super().__init__(file)
        self.entry_to_key_mapper = entry_to_key_mapper or self.DEFAULT_ENTRY_TO_KEY_MAPPER
        self.results             = self.__init_or_load(clue_count, working_df, models)

    def __init_or_load(self, clue_count, working_df, models):
        if os.path.exists(self.file):
            results = self.load()
            if models is not None:
                results = {
                    model_key: model_results
                    for model_key, model_results in results.items()
                    if model_key in models
                }
                if working_df is not None:
                    results.update({
                        model_key: {
                            self.entry_to_key_mapper(entry): {
                                str(i): { 'simple': None, 'complex': None }
                                for i in range(1, clue_count+1)
                            }
                            for _, entry in working_df['data'].iterrows()
                        }
                        for model_key in models
                        if model_key not in results or results[model_key] is None
                    })

        else:
            results = {
                model_key: {
                    self.entry_to_key_mapper(entry): {
                        str(i): { 'simple': None, 'complex': None }
                        for i in range(1, clue_count+1)
                    }
                    for _, entry in working_df['data'].iterrows()
                }
                for model_key in models
            }

        return results

    def encode(self, entry):
        return self.entry_to_key_mapper(entry)

class ExtractionResult(Result):

    def __init__(self, file, topics=None, models=None) -> None:
        super().__init__(file)
        self.results = self.__init_or_load(topics, models)

    def __init_or_load(self, topics, models):
        if os.path.exists(self.file):
            results = self.load()
            if topics is not None:
                if models is not None:
                    for topic_results in results.values():
                        topic_results.update({
                            model_key: [] for model_key in models
                            if model_key not in topic_results
                        })
                    results.update({
                        topic: { model_key: [] for model_key in models }
                        for topic in topics if topic not in results
                    })
        else:
            results = {
                topic: { model_key: [] for model_key in models }
                for topic in topics
            }
        return results
