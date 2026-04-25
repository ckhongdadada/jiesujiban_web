import pathlib
pathlib.Path('_debug.txt').write_text('hello from test', encoding='utf-8')
print('file written')
