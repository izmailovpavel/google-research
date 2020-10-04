import tabulate


def make_table(tabulate_dict, iteration, header_freq):
  table = tabulate.tabulate(
      [tabulate_dict.values()], tabulate_dict.keys(), tablefmt='simple',
      floatfmt='8.7f')
  if iteration % header_freq == 0:
    table = table.split('\n')
    table = '\n'.join([table[1]] + table)
  else:
    table = table.split('\n')[2]
  return table