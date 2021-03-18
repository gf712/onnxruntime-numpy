import re
import string

# obtained and modified from https://github.com/tensorflow/tensorflow/blob/590d6eef7e91a6a7392c8ffffb7b58f2e0c8bc6b/tensorflow/python/ops/special_math_ops.py#L311
def einsum_parse_and_resolve_equation(equation, input_shapes):
    """Helper for einsum() that splits/resolves inputs & outputs.
    Args:
      equation: Equation string given as argument to einsum().
      input_shapes: List of the shapes of all inputs given to einsum()
    Returns:
      input_axis_labels, output_axis_labels where:
        input_axis_labels: List of length len(input_shapes) of strings
        representing the character label for each dimension of each given input,
        resolving any broadcast (...) axes,
      output_axis_labels: A string of character labels for each axes of output
        tensor, filling in missing output subscripts and broadcast axes.
    Raises:
      ValueError: If equation is in the uncorrect format, incorrect number of
        inputs given or broadcast axes "..." or output axes could not be resolved.
    """
    equation = equation.replace(' ', '')
    match = re.match('^([a-zA-Z,.]+)(->[a-zA-Z.]*)?$', equation)
    if not match:
        raise ValueError('Indices have incorrect format: %s' % equation)

    input_axis_labels = match.group(1).split(',')
    output_axis_labels = match.group(2)[2:] if match.group(2) else None

    if len(input_shapes) != len(input_axis_labels):
        raise ValueError('Got %d arguments for equation "%s", expecting %d' %
                         (len(input_shapes), equation, len(input_axis_labels)))

    # Resolve Ellipsis
    # Assign axes labels for unspecified dimensions in inputs. Labels taken
    # from unused labels. Follow numpy einsum broadcasting conventions for
    # tensors of different length and unlabeled output.
    ellipsis_axes = ''
    if '...' in equation:
        unused = ''.join([c for c in string.ascii_letters
                          if c not in ''.join(input_axis_labels)])
        for i, ax in enumerate(input_axis_labels):
            if '...' in ax:
                parts = ax.split('...')
                if len(parts) != 2:
                    raise ValueError('Unable to resolve ellipsis. Excess number found.')
                n = len(input_shapes[i]) - len(''.join(parts))
                if n < 0:
                    raise ValueError('Ellipses lengths do not match.')
                if len(unused) < n:
                    raise ValueError(
                        'Unable to resolve ellipsis, too many distinct labels.')
                replace_axes = unused[-n:] if n > 0 else ''
                input_axis_labels[i] = input_axis_labels[i].replace('...',
                                                                    replace_axes)
                if len(replace_axes) > len(ellipsis_axes):
                    ellipsis_axes = replace_axes

        if any(['.' in ax for ax in input_axis_labels]):
            raise ValueError('period "." found outside of ellipsis')

        if output_axis_labels is not None:
            output_axis_labels = output_axis_labels.replace('...', ellipsis_axes)
            if '.' in output_axis_labels:
                raise ValueError('period "." found outside of ellipsis')

    if output_axis_labels is None:
        # infer the output subscripts if not given, assume alphabetical order,
        # but always place ellipsis axes before given.
        axis_labels = set(''.join(input_axis_labels)) - set(ellipsis_axes)
        indices = ''.join(sorted(axis_labels))
        counts = {ax: 0 for ax in indices}
        for axes_ in input_axis_labels:
            for ax in axes_:
                if ax not in ellipsis_axes:
                    counts[ax] += 1

        output_axis_labels = ellipsis_axes + ''.join(
            sorted(ax for ax in axis_labels if counts[ax] == 1))

    return input_axis_labels, output_axis_labels
