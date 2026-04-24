import sys
import pathlib
import traceback

sys.path.insert(0, '.')
lines = []

try:
    from tools.optuna_search_classifier import parse_args, setup_logger
    lines.append('import ok')
except Exception as e:
    lines.append(f'import fail: {e}')
    lines.append(traceback.format_exc())

try:
    import optuna
    lines.append(f'optuna ok: {optuna.__version__}')
except Exception as e:
    lines.append(f'optuna fail: {e}')

try:
    from training.train_unit_classifier_fgm import Config, BertClassifier, DataProcessor, TextDataset, build_arg_parser
    lines.append('training imports ok')
except Exception as e:
    lines.append(f'training import fail: {e}')
    lines.append(traceback.format_exc())

try:
    args = parse_args()
    lines.append(f'parse_args ok: n_trials={args.n_trials}, output_dir={args.output_dir}')
except Exception as e:
    lines.append(f'parse_args fail: {e}')
    lines.append(traceback.format_exc())

pathlib.Path('_debug.txt').write_text('\n'.join(lines), encoding='utf-8')
print('debug written')
