import argparse
import re
import pandas as pd
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser(
        "Prints a table from metrics files\n"
    )
    parser.add_argument(
        '--config', '-c', type=str, default='config.csv',
        help='Path to the config csv with `name` and `path` columns. '
             '`name` is a model name, and '
             '`path` is a path to metrics file`'
    )
    parser.add_argument(
        '--extension', '-e', type=str,
        choices=['html', 'latex', 'csv'],
        default='csv',
        help='Format of a table'
    )
    parser.add_argument(
        '--output', '-o', type=str,
        default='output.csv',
        help='Path to the output table'
    )
    parser.add_argument(
        '--precision', '-p', type=int,
        default=4, help='Precision in final table'
    )
    return parser


def format_result(result, fmt=None):
    if np.isnan(result['std']):
        result_str = str(result['mean'])
    else:
        result_str = f"{result['mean']}±{result['std']}"
    if fmt is not None:
        result_str = fmt.format(result_str)
    return result_str


if __name__ == "__main__":
    parser = get_parser()
    config, unknown = parser.parse_known_args()
    if len(unknown) != 0:
        raise ValueError("Unknown argument " + unknown[0])

    metrics = []
    models = pd.read_csv(config.config)
    for path, name in zip(models['path'], models['name']):
        parsed_metrics = pd.read_csv(path, header=None)
        parsed_metrics = {x[1][0]: x[1][1]
                          for x in parsed_metrics.iterrows()}
        parsed_metrics['Model'] = name
        metrics.append(parsed_metrics)
    metrics = pd.DataFrame(metrics)
    metrics = metrics.groupby('Model', sort=False)
    metrics = metrics.agg([np.mean, np.std]).reset_index()
    metrics = metrics.rename(columns={'valid': 'Valid',
                                      'unique@1000': 'Unique@1k',
                                      'unique@10000': 'Unique@10k'})
    targets = ['Model', 'Valid', 'Unique@1k',
               'Unique@10k', 'FCD/Test', 'FCD/TestSF',
               'SNN/Test', 'SNN/TestSF', 'Frag/Test',
               'Frag/TestSF', 'Scaf/Test', 'Scaf/TestSF',
               'IntDiv', 'IntDiv2', 'Filters', 'Novelty']
    directions = [2, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    metrics = metrics[targets]

    bf_pattern = {
        'csv': '{}',
        'html': '<b>{}</b>',
        'latex': r'!bf1! {} !bf2!'
    }[config.extension]
    it_pattern = {
        'csv': '{}',
        'html': '<i>{}</i>',
        'latex': r'!it1! {} !it2!'
    }[config.extension]

    arrow = {
        'csv': [' (↓)', ' (↑)', ''],
        'html': [' (↓)', ' (↑)', ''],
        'latex': [r' ($\downarrow$)', r' ($\uparrow$)', '']
    }[config.extension]

    metrics = metrics.set_index('Model')
    models = [x for x in metrics.index if x != 'Train']

    if 'Train' in metrics.index:
        models = ['Train'] + models
    metrics = metrics.loc[models].reset_index()

    for col, d in zip(targets[1:], directions[1:]):
        metrics[col] = metrics[col] \
            .astype(float) \
            .round(config.precision)
        sgn = (2 * d - 1)
        max_val = sgn * np.max(
            [sgn * m for m, n in zip(metrics[col]['mean'],
                                     metrics['Model'])
             if n != 'Train'])
        metric = [format_result(x)
                  if x['mean'] != max_val or n == 'Train'
                  else format_result(x, bf_pattern)
                  for (_, x), n in zip(metrics[col].iterrows(),
                                       metrics['Model'])]
        metrics = metrics.drop(col, axis=1, level=0)
        metrics[col] = metric
    if 'Train' in models:
        metrics.iloc[0] = metrics.iloc[0].apply(it_pattern.format)

    metrics = metrics.round(config.precision)
    metrics.columns = metrics.columns.droplevel(1)
    if config.extension == 'csv':
        metrics.to_csv(config.output, index=None)
    elif config.extension == 'html':
        html = metrics.to_html(index=None)
        html = re.sub('&lt;', '<', html)
        html = re.sub('&gt;', '>', html)
        header, footer = html.split('</thead>')
        header += '</thead>'
        header = header.split('\n')
        values = [x.strip()[4:-5]
                  for x in header[3:-2]]
        spans = ['rowspan' if '/' not in x else 'colspan'
                 for x in values]
        first_header = [x.split('/')[0] for x in values]
        second_header = [x.split('/')[1] for x in values
                         if '/' in x]
        new_header = header[:3]
        i = 0
        total = 0
        while i < len(first_header):
            h = first_header[i]
            new_header.append(
                ' ' * 6 + '<th {}="2">{}{}</th>'.format(
                    spans[i], h, arrow[directions[i]]
                )
            )
            i += 1
            total += 1
            while i < len(first_header) - 1 and first_header[i] == h:
                i += 1
        new_header.extend(['    </tr>',
                           '    <tr>'])
        for h in second_header:
            new_header.append(' ' * 6 + '<th>{}</th>'.format(h))
        new_header.extend(header[-2:])
        header = '\n'.join(new_header)
        html = header + footer
        html = ('<html>\n<head>\n<meta charset="utf-8">\n</head>\n' +
                html +
                '\n</html>')
        with open(config.output, 'w', encoding='utf-8') as f:
            f.write(html)
    elif config.extension == 'latex':
        latex = metrics.to_latex(index=None)
        latex = latex.split('\n')
        header1 = [r'\multirow{2}{*}{' + x.strip() + arrow[d] + '} '
                   if '/' not in x
                   else r'\multicolumn{2}{c}{' + x.split('/')[0].strip() +
                        arrow[d] + '}'
                   for x, d in zip(latex[2].strip()[:-2].split('&'),
                                   directions)]
        header1 = [x for i, x in enumerate(header1)
                   if i == 0 or header1[i - 1] != header1[i]]
        header2 = [x.split('/')[1].strip()
                   if '/' in x
                   else ''
                   for x, d in zip(latex[2].strip()[:-2].split('&'),
                                   directions)]

        latex[2] = ' & '.join(header1) + r'\\'
        latex.insert(3, ' & '.join(header2) + r'\\')
        latex = '\n'.join(latex)
        latex = re.sub(r'!bf1!', r'{\\bf', latex)
        latex = re.sub(r'!bf2!', '}', latex)
        latex = re.sub(r'!it1!', r'{\\it', latex)
        latex = re.sub(r'!it2!', '}', latex)
        with open(config.output, 'w', encoding='utf-8') as f:
            f.write(latex)
