import json

def jprint(*args, indent=2, **kwargs):
    return print(*[ json.dumps(obj, ensure_ascii=False, indent=indent) for obj in args ], **kwargs)

def read_json(path):
    with open(path, 'r', encoding='utf-8') as ifile:
        return json.load(ifile)

def write_json(path, data, indent=2, ensure_ascii=True, **kwargs):
    with open(path, 'w', encoding='utf-8') as ofile:
        json.dump(
            data, ofile, indent=indent,
            ensure_ascii=ensure_ascii, **kwargs
        )
