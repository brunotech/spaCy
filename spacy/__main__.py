# coding: utf8
from __future__ import print_function

# NB! This breaks in plac on Python 2!!
# from __future__ import unicode_literals

if __name__ == "__main__":
    import plac
    import sys
    from wasabi import msg
    from spacy.cli import download, link, info, package, train, pretrain, convert
    from spacy.cli import init_model, profile, evaluate, validate, debug_data

    commands = {
        "download": download,
        "link": link,
        "info": info,
        "train": train,
        "pretrain": pretrain,
        "debug-data": debug_data,
        "evaluate": evaluate,
        "convert": convert,
        "package": package,
        "init-model": init_model,
        "profile": profile,
        "validate": validate,
    }
    if len(sys.argv) == 1:
        msg.info("Available commands", ", ".join(commands), exits=1)
    command = sys.argv.pop(1)
    sys.argv[0] = f"spacy {command}"
    if command in commands:
        plac.call(commands[command], sys.argv[1:])
    else:
        available = f'Available: {", ".join(commands)}'
        msg.fail(f"Unknown command: {command}", available, exits=1)
