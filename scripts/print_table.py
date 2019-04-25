import pandas as pd
from collections import OrderedDict
import argparse
import numpy as np
import re


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


if __name__ == "__main__":
    parser = get_parser()
    config, unknown = parser.parse_known_args()
    if len(unknown) != 0:
        raise ValueError("Unknown argument " + unknown[0])

    metrics = OrderedDict()
    models = pd.read_csv(config.config)
    for path, name in zip(models['path'], models['name']):
        metrics[name] = pd.read_csv(path, header=None)
        metrics[name] = {x[1][0]: x[1][1]
                         for x in metrics[name].iterrows()}
        metrics[name]['Model'] = name
    metrics = pd.DataFrame(metrics).T
    metrics = metrics.rename(columns={'valid': 'Valid',
                                      'unique@1000': 'Unique@1k',
                                      'unique@10000': 'Unique@10k'})
    targets = ['Model', 'Valid', 'Unique@1k',
               'Unique@10k', 'FCD/Test', 'FCD/TestSF',
               'SNN/Test', 'SNN/TestSF', 'Frag/Test',
               'Frag/TestSF', 'Scaf/Test', 'Scaf/TestSF',
               'IntDiv', 'IntDiv2', 'Filters']
    directions = [2, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
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

    for col, d in zip(targets[1:], directions[1:]):
        metrics[col] = metrics[col] \
            .astype(float) \
            .round(config.precision)
        max_val = (2 * d - 1) * np.max(
            [(2 * d - 1) * m for m, n in zip(metrics[col],
                                             metrics['Model'])
             if n != 'Train'])
        metrics[col] = [str(x) if x != max_val or n == 'Train'
                        else bf_pattern.format(x)
                        for x, n in zip(metrics[col],
                                        metrics['Model'])]
    for col in targets[::-1]:
        metrics[col] = [it_pattern.format(x)
                        if n == 'Train' else x
                        for x, n in zip(metrics[col],
                                        metrics['Model'])]

    metrics = metrics.round(config.precision)
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
                    spans[i], h, arrow[directions[total]]
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
        with open(config.output, 'w') as f:
            f.write(latex)
